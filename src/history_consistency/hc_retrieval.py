# src/history_consistency/hc_retrieval.py

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

from history_consistency.hc_config import (
    HC_DIMENSION_QUERY,
    HC_KEYWORDS_CORE,
    HC_KEYWORDS_TEMPORAL,
    HC_KEYWORDS_DOCUMENT_CONTEXT,
    HC_NEGATIVE_TERMS,
    SOURCE_PRIOR,
    SECTION_PRIOR,
    PREFERRED_SECTIONS,
    BASE_EVIDENCE_SCORE_WEIGHTS,
    TOP_K_EVIDENCE,
    TOP_K_EVIDENCE_PER_YEAR,
)
from history_consistency.hc_loader import CompanyEvidencePool


@dataclass(frozen=True)
class RetrievalResult:
    """
    Retrieval result for one company.
    """

    company: str
    candidates: pd.DataFrame
    evidence_count: int
    selected_count: int


def normalize_text_for_match(text: str) -> str:
    """
    Lightweight text normalization for keyword matching.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def phrase_count(text: str, phrase: str) -> int:
    """
    Count phrase occurrences with simple boundary-aware matching.

    For multi-word phrases, this prevents substring matching from being too loose.
    """
    text_norm = normalize_text_for_match(text)
    phrase_norm = normalize_text_for_match(phrase)

    if not phrase_norm:
        return 0

    escaped = re.escape(phrase_norm)
    pattern = rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])"
    return len(re.findall(pattern, text_norm))


def keyword_relevance_score(
    text: str,
    core_keywords: Sequence[str] = HC_KEYWORDS_CORE,
    temporal_keywords: Sequence[str] = HC_KEYWORDS_TEMPORAL,
    document_keywords: Sequence[str] = HC_KEYWORDS_DOCUMENT_CONTEXT,
) -> float:
    """
    Compute HC keyword relevance in [0, 1].

    HC requires:
    - purpose / narrative consistency language
    - temporal continuity language
    - disclosure / document context language

    Core keywords are weighted most heavily.
    """
    text_norm = normalize_text_for_match(text)

    if not text_norm:
        return 0.0

    core_hits = sum(1 for kw in core_keywords if phrase_count(text_norm, kw) > 0)
    temporal_hits = sum(1 for kw in temporal_keywords if phrase_count(text_norm, kw) > 0)
    document_hits = sum(1 for kw in document_keywords if phrase_count(text_norm, kw) > 0)

    core_ratio = core_hits / max(len(core_keywords), 1)
    temporal_ratio = temporal_hits / max(len(temporal_keywords), 1)
    document_ratio = document_hits / max(len(document_keywords), 1)

    raw_score = (
        0.50 * core_ratio
        + 0.35 * temporal_ratio
        + 0.15 * document_ratio
    )

    # Light boost if the text contains both purpose language and temporal language.
    if core_hits > 0 and temporal_hits > 0:
        raw_score += 0.15

    return float(min(max(raw_score, 0.0), 1.0))


def context_completeness_score(row: pd.Series) -> float:
    """
    Compute context completeness in [0, 1].

    This checks whether a chunk has enough metadata and enough textual detail
    to support HC scoring.
    """
    score = 0.0

    text = str(row.get("text_clean", "") or "")
    source = str(row.get("source", "") or "").lower()
    section = str(row.get("section", "") or "").lower()
    year = row.get("year", None)

    token_count = row.get("token_count", None)
    char_count = row.get("char_count", None)

    # Text detail.
    if len(text) >= 300:
        score += 0.25
    elif len(text) >= 150:
        score += 0.15
    elif len(text) >= 80:
        score += 0.08

    # Year is important for HC.
    if pd.notna(year):
        score += 0.20

    # Source is available.
    if source:
        score += 0.15

    # Section is available.
    if section:
        score += 0.15

    # Preferred section.
    if any(sec in section for sec in PREFERRED_SECTIONS):
        score += 0.15

    # Reasonable token / character count.
    if pd.notna(token_count):
        try:
            tc = float(token_count)
            if 80 <= tc <= 500:
                score += 0.10
            elif 40 <= tc < 80:
                score += 0.05
        except Exception:
            pass
    elif pd.notna(char_count):
        try:
            cc = float(char_count)
            if 400 <= cc <= 3000:
                score += 0.10
            elif 200 <= cc < 400:
                score += 0.05
        except Exception:
            pass

    return float(min(max(score, 0.0), 1.0))


def negative_term_penalty(
    text: str,
    negative_terms: Sequence[str] = HC_NEGATIVE_TERMS,
) -> float:
    """
    Return a multiplicative penalty for likely boilerplate / irrelevant sections.

    Output range:
    - 1.00 means no penalty
    - lower values mean more penalty
    """
    text_norm = normalize_text_for_match(text)

    if not text_norm:
        return 1.0

    hit_count = sum(1 for term in negative_terms if phrase_count(text_norm, term) > 0)

    if hit_count == 0:
        return 1.0
    if hit_count == 1:
        return 0.90
    if hit_count == 2:
        return 0.80

    return 0.70


def get_source_prior(source: str) -> float:
    """
    Return HC source prior.

    Source prior is used for ranking only, not final scoring.
    """
    source_norm = str(source or "").lower().strip()
    return float(SOURCE_PRIOR.get(source_norm, 1.0))


def get_section_prior(section: str) -> float:
    """
    Return HC section prior.

    If exact key is unavailable, use substring matching.
    """
    section_norm = str(section or "").lower().strip()

    if not section_norm:
        return 1.0

    if section_norm in SECTION_PRIOR:
        return float(SECTION_PRIOR[section_norm])

    for key, value in SECTION_PRIOR.items():
        if key in section_norm:
            return float(value)

    return 1.0


def compute_tfidf_similarity(
    texts: Sequence[str],
    query: str = HC_DIMENSION_QUERY,
) -> np.ndarray:
    """
    Compute TF-IDF cosine similarity between each text and the HC dimension query.

    Returns:
    np.ndarray of shape (len(texts),), values in [0, 1] approximately.
    """
    clean_texts = [str(t or "") for t in texts]

    if len(clean_texts) == 0:
        return np.array([])

    if TfidfVectorizer is None or cosine_similarity is None:
        return np.zeros(len(clean_texts), dtype=float)

    corpus = [query] + clean_texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )

    try:
        matrix = vectorizer.fit_transform(corpus)
        query_vec = matrix[0]
        text_vecs = matrix[1:]
        sims = cosine_similarity(text_vecs, query_vec).reshape(-1)
    except ValueError:
        sims = np.zeros(len(clean_texts), dtype=float)

    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
    sims = np.clip(sims, 0.0, 1.0)

    return sims


def compute_base_evidence_score(
    keyword_relevance: float,
    context_completeness: float,
    rag_similarity: float,
) -> float:
    """
    Base evidence score in [0, 1].

    Formula is aligned with PA logic:
    0.40 × keyword_relevance
    + 0.35 × context_completeness
    + 0.25 × rag_similarity
    """
    w = BASE_EVIDENCE_SCORE_WEIGHTS

    score = (
        w["keyword_relevance"] * keyword_relevance
        + w["context_completeness"] * context_completeness
        + w["rag_similarity"] * rag_similarity
    )

    return float(min(max(score, 0.0), 1.0))


def score_company_candidates(pool: CompanyEvidencePool) -> pd.DataFrame:
    """
    Score all candidate chunks within one company evidence pool.

    Output columns include:
    - hc_keyword_relevance
    - hc_context_completeness
    - hc_rag_similarity
    - hc_base_evidence_score_0_1
    - hc_source_prior
    - hc_section_prior
    - hc_negative_penalty
    - hc_rank_score
    """
    df = pool.data.copy()

    if df.empty:
        return df

    df["hc_keyword_relevance"] = df["text_clean"].apply(keyword_relevance_score)
    df["hc_context_completeness"] = df.apply(context_completeness_score, axis=1)

    tfidf_sims = compute_tfidf_similarity(df["text_clean"].tolist())
    df["hc_rag_similarity"] = tfidf_sims

    df["hc_base_evidence_score_0_1"] = df.apply(
        lambda row: compute_base_evidence_score(
            keyword_relevance=float(row["hc_keyword_relevance"]),
            context_completeness=float(row["hc_context_completeness"]),
            rag_similarity=float(row["hc_rag_similarity"]),
        ),
        axis=1,
    )

    df["hc_source_prior"] = df["source"].apply(get_source_prior)
    df["hc_section_prior"] = df["section"].apply(get_section_prior)
    df["hc_negative_penalty"] = df["text_clean"].apply(negative_term_penalty)

    # Ranking score only.
    # Source and section prior help select better evidence but should not directly
    # inflate final LLM rubric score.
    df["hc_rank_score"] = (
        df["hc_base_evidence_score_0_1"]
        * df["hc_source_prior"]
        * df["hc_section_prior"]
        * df["hc_negative_penalty"]
    )

    df["hc_rank_score"] = df["hc_rank_score"].clip(lower=0.0)

    return df.sort_values(
        ["hc_rank_score", "hc_base_evidence_score_0_1"],
        ascending=[False, False],
    ).reset_index(drop=True)


def select_top_hc_evidence(
    scored_df: pd.DataFrame,
    top_k: int = TOP_K_EVIDENCE,
    top_k_per_year: int = TOP_K_EVIDENCE_PER_YEAR,
) -> pd.DataFrame:
    """
    Select top HC evidence with a per-year cap.

    This avoids one year dominating the HC evidence pack.
    """
    if scored_df.empty:
        return scored_df.copy()

    df = scored_df.copy()
    selected_parts: List[pd.DataFrame] = []

    # Rows with valid years: apply per-year cap.
    valid_year_df = df[df["year"].notna()].copy()
    missing_year_df = df[df["year"].isna()].copy()

    if not valid_year_df.empty:
        valid_year_df["year_int"] = valid_year_df["year"].astype(int)

        for _, year_group in valid_year_df.groupby("year_int"):
            selected_parts.append(
                year_group.sort_values("hc_rank_score", ascending=False)
                .head(top_k_per_year)
            )

    # Missing-year rows can still be useful, but they should not dominate.
    if not missing_year_df.empty:
        selected_parts.append(
            missing_year_df.sort_values("hc_rank_score", ascending=False)
            .head(max(1, top_k_per_year))
        )

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = df.head(0).copy()

    selected = selected.sort_values(
        ["hc_rank_score", "hc_base_evidence_score_0_1"],
        ascending=[False, False],
    ).head(top_k)

    selected = selected.reset_index(drop=True)
    selected["hc_selected_rank"] = selected.index + 1

    return selected


def retrieve_company_hc_candidates(
    pool: CompanyEvidencePool,
    top_k: int = TOP_K_EVIDENCE,
    top_k_per_year: int = TOP_K_EVIDENCE_PER_YEAR,
) -> RetrievalResult:
    """
    Score and select HC evidence for one company.
    """
    scored = score_company_candidates(pool)
    selected = select_top_hc_evidence(
        scored_df=scored,
        top_k=top_k,
        top_k_per_year=top_k_per_year,
    )

    return RetrievalResult(
        company=pool.company,
        candidates=selected,
        evidence_count=len(scored),
        selected_count=len(selected),
    )


def retrieve_all_company_hc_candidates(
    pools: Dict[str, CompanyEvidencePool],
    top_k: int = TOP_K_EVIDENCE,
    top_k_per_year: int = TOP_K_EVIDENCE_PER_YEAR,
) -> Dict[str, RetrievalResult]:
    """
    Run HC retrieval for all company evidence pools.
    """
    results: Dict[str, RetrievalResult] = {}

    for company, pool in pools.items():
        results[company] = retrieve_company_hc_candidates(
            pool=pool,
            top_k=top_k,
            top_k_per_year=top_k_per_year,
        )

    return results


def flatten_retrieval_results(
    results: Dict[str, RetrievalResult],
) -> pd.DataFrame:
    """
    Convert retrieval results dict into one dataframe.
    """
    frames: List[pd.DataFrame] = []

    for company, result in results.items():
        if result.candidates.empty:
            continue

        df = result.candidates.copy()
        df["company"] = company
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    from history_consistency.hc_loader import load_company_evidence_pools

    pools = load_company_evidence_pools()
    results = retrieve_all_company_hc_candidates(pools)
    flat = flatten_retrieval_results(results)

    print("Companies:", len(results))
    print("Selected evidence rows:", len(flat))

    if not flat.empty:
        preview_cols = [
            "company",
            "year",
            "source",
            "section",
            "hc_rank_score",
            "hc_keyword_relevance",
            "hc_context_completeness",
            "hc_rag_similarity",
            "chunk_id",
        ]
        available_cols = [c for c in preview_cols if c in flat.columns]
        print(flat[available_cols].head(20).to_string(index=False))
