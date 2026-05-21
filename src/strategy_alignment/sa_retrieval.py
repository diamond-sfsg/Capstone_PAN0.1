# src/strategy_alignment/sa_retrieval.py

"""
Candidate retrieval for Strategy & Source Alignment.

This module retrieves and ranks candidate evidence for each SA question.

Retrieval is not final scoring.
It prepares top evidence for LLM evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

from .sa_config import (
    SA_QUESTIONS,
    QUESTION_ORDER,
    SA_QUESTION_QUERIES,
    TOP_K_EVIDENCE_PER_QUESTION,
    MIN_EVIDENCE_PER_QUESTION,
    SA_SOURCE_PRIOR,
    SA_SECTION_PRIOR,
)
from .sa_evidence_score import add_evidence_scores, clip, normalize_text


# ============================================================
# TF-IDF retrieval
# ============================================================

def compute_tfidf_similarity(
    texts: List[str],
    query: str,
) -> List[float]:
    """
    Compute TF-IDF cosine similarity between each text and one question query.

    Returns normalized scores in [0, 1].
    """
    if not texts:
        return []

    if TfidfVectorizer is None or cosine_similarity is None:
        # Fallback if sklearn is unavailable.
        return [0.0] * len(texts)

    cleaned_texts = [normalize_text(text) for text in texts]
    cleaned_query = normalize_text(query)

    if not cleaned_query:
        return [0.0] * len(texts)

    corpus = cleaned_texts + [cleaned_query]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        matrix = vectorizer.fit_transform(corpus)

        text_matrix = matrix[:-1]
        query_vector = matrix[-1]

        scores = cosine_similarity(text_matrix, query_vector).flatten().tolist()

    except Exception:
        return [0.0] * len(texts)

    if not scores:
        return [0.0] * len(texts)

    max_score = max(scores)
    if max_score <= 0:
        return [0.0] * len(texts)

    # Normalize within the company evidence pool.
    return [clip(score / max_score, 0.0, 1.0) for score in scores]


# ============================================================
# Ranking priors
# ============================================================

def get_source_prior(source: Any) -> float:
    value = normalize_text(source)
    return float(SA_SOURCE_PRIOR.get(value, SA_SOURCE_PRIOR["unknown"]))


def get_section_prior(section: Any) -> float:
    value = normalize_text(section)

    if value in SA_SECTION_PRIOR:
        return float(SA_SECTION_PRIOR[value])

    # Soft matching for messy section labels.
    for key, prior in SA_SECTION_PRIOR.items():
        if key != "unknown" and key in value:
            return float(prior)

    return float(SA_SECTION_PRIOR["unknown"])


def compute_rank_score(row: pd.Series) -> float:
    """
    Rank candidates using evidence base score and priors.

    Source and section priors only affect retrieval ranking, not final score.
    """
    base = float(row.get("base_evidence_score_0_1", 0.0))
    source_prior = float(row.get("source_prior", 1.0))
    section_prior = float(row.get("section_prior", 1.0))

    return base * source_prior * section_prior


# ============================================================
# Candidate retrieval
# ============================================================

def retrieve_question_candidates(
    company_chunks: pd.DataFrame,
    question_id: str,
    purpose_reference: Optional[Dict[str, Any]] = None,
    top_k: int = TOP_K_EVIDENCE_PER_QUESTION,
) -> pd.DataFrame:
    """
    Retrieve top candidate evidence for one company and one SA question.
    """
    if question_id not in SA_QUESTIONS:
        raise KeyError(f"Unknown SA question_id: {question_id}")

    if company_chunks.empty:
        return _empty_candidate_frame(question_id)

    if "text_clean" not in company_chunks.columns:
        raise ValueError("company_chunks must contain column: text_clean")

    query = SA_QUESTION_QUERIES[question_id]

    working = company_chunks.copy()
    working["text_clean"] = working["text_clean"].fillna("").astype(str).str.strip()
    working = working[working["text_clean"].ne("")].copy()

    if working.empty:
        return _empty_candidate_frame(question_id)

    rag_scores = compute_tfidf_similarity(
        texts=working["text_clean"].tolist(),
        query=query,
    )

    scored = add_evidence_scores(
        working,
        question_id=question_id,
        rag_scores=rag_scores,
        purpose_reference=purpose_reference,
    )

    scored["source_prior"] = scored["source"].map(get_source_prior)
    scored["section_prior"] = scored["section"].map(get_section_prior)

    scored["rank_score"] = scored.apply(compute_rank_score, axis=1)

    scored = scored.sort_values(
        by=[
            "rank_score",
            "base_evidence_score_0_1",
            "rag_similarity",
            "keyword_relevance",
            "context_completeness",
        ],
        ascending=False,
    ).reset_index(drop=True)

    scored["rank"] = scored.index + 1
    scored["question_name"] = SA_QUESTIONS[question_id]

    return scored.head(top_k).reset_index(drop=True)


def retrieve_company_candidates(
    company_chunks: pd.DataFrame,
    purpose_reference: Dict[str, Any],
    top_k: int = TOP_K_EVIDENCE_PER_QUESTION,
) -> Dict[str, pd.DataFrame]:
    """
    Retrieve candidates for all SA questions for one company.
    """
    output = {}

    for question_id in QUESTION_ORDER:
        output[question_id] = retrieve_question_candidates(
            company_chunks=company_chunks,
            question_id=question_id,
            purpose_reference=purpose_reference,
            top_k=top_k,
        )

    return output


def retrieve_company_candidates_long(
    company: str,
    company_chunks: pd.DataFrame,
    purpose_reference: Dict[str, Any],
    top_k: int = TOP_K_EVIDENCE_PER_QUESTION,
) -> pd.DataFrame:
    """
    Retrieve candidates for all SA questions and return one long DataFrame.
    """
    candidate_map = retrieve_company_candidates(
        company_chunks=company_chunks,
        purpose_reference=purpose_reference,
        top_k=top_k,
    )

    frames = []

    for question_id, df in candidate_map.items():
        if df.empty:
            continue

        temp = df.copy()
        temp["company"] = company
        temp["purpose_statement_normalized"] = purpose_reference.get(
            "purpose_statement_normalized",
            "",
        )
        temp["extracted_purpose"] = purpose_reference.get("extracted_purpose", "")
        temp["purpose_statement_raw"] = purpose_reference.get("purpose_statement_raw", "")
        temp["purpose_confidence_0_1"] = purpose_reference.get(
            "purpose_confidence_0_1",
            0.0,
        )
        frames.append(temp)

    if not frames:
        return _empty_candidate_frame()

    out = pd.concat(frames, ignore_index=True)

    # Keep output readable and stable.
    preferred_columns = [
        "company",
        "question_id",
        "question_name",
        "rank",
        "rank_score",
        "chunk_id",
        "year",
        "source",
        "section",
        "text_clean",
        "keyword_relevance",
        "context_completeness",
        "rag_similarity",
        "base_evidence_score_0_1",
        "evidence_quality_factor",
        "overlap_label",
        "overlap_factor",
        "source_prior",
        "section_prior",
        "extracted_purpose",
        "purpose_statement_normalized",
        "purpose_statement_raw",
        "purpose_confidence_0_1",
    ]

    existing = [col for col in preferred_columns if col in out.columns]
    remaining = [col for col in out.columns if col not in existing]

    return out[existing + remaining].reset_index(drop=True)


# ============================================================
# Diagnostics
# ============================================================

def summarize_candidate_coverage(
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize how many candidate evidence rows each company/question has.
    """
    if candidates_df.empty:
        return pd.DataFrame(
            columns=[
                "company",
                "question_id",
                "num_candidates",
                "meets_min_evidence",
            ]
        )

    summary = (
        candidates_df.groupby(["company", "question_id"], dropna=False)
        .agg(
            num_candidates=("chunk_id", "count"),
            max_rank_score=("rank_score", "max"),
            mean_rank_score=("rank_score", "mean"),
            max_base_evidence_score=("base_evidence_score_0_1", "max"),
        )
        .reset_index()
    )

    summary["meets_min_evidence"] = (
        summary["num_candidates"] >= MIN_EVIDENCE_PER_QUESTION
    )

    return summary


def _empty_candidate_frame(question_id: Optional[str] = None) -> pd.DataFrame:
    columns = [
        "company",
        "question_id",
        "question_name",
        "rank",
        "rank_score",
        "chunk_id",
        "year",
        "source",
        "section",
        "text_clean",
        "keyword_relevance",
        "context_completeness",
        "rag_similarity",
        "base_evidence_score_0_1",
        "evidence_quality_factor",
        "overlap_label",
        "overlap_factor",
        "source_prior",
        "section_prior",
        "extracted_purpose",
        "purpose_statement_normalized",
        "purpose_statement_raw",
        "purpose_confidence_0_1",
    ]

    df = pd.DataFrame(columns=columns)

    if question_id:
        df["question_id"] = pd.Series(dtype="object")
        df["question_name"] = pd.Series(dtype="object")

    return df
