from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

from purpose_articulation.pa_config import (
    TOP_K_CANDIDATES_PER_QUESTION,
    TOP_K_EVIDENCE_PER_SOURCE_Q3,
    TOP_K_GLOBAL_EVIDENCE_Q3,
    TOP_K_LLM_EVIDENCE_Q1_Q2,
)
from purpose_articulation.pa_config import (
    SECTION_BLACKLIST,
    PA_POSITIVE_TERMS,
    PA_NEGATIVE_TERMS,
    PA_CANDIDATE_WEIGHTS,
    NEGATIVE_BOILERPLATE_PENALTY,
)
from purpose_articulation.pa_evidence_score import (
    compute_base_evidence_score,
    context_completeness_score,
    evidence_quality_factor,
    get_source_prior,
    keyword_relevance_score,
    rule_based_pa_tone_bonus,
)
from purpose_articulation.pa_rubric import PA_QUESTIONS


def compute_tfidf_similarity(texts: list[str], query: str) -> list[float]:
    """
    Compute TF-IDF cosine similarity between each evidence text and one query.

    This acts as the sparse RAG / query matching signal.
    """
    clean_texts = [str(text) if text is not None else "" for text in texts]

    if not clean_texts:
        return []

    corpus = clean_texts + [query]

    if TfidfVectorizer is not None and cosine_similarity is not None:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
            )

            matrix = vectorizer.fit_transform(corpus)

            doc_matrix = matrix[:-1]
            query_matrix = matrix[-1]

            scores = cosine_similarity(doc_matrix, query_matrix).ravel()

            return [float(score) for score in scores]

        except ValueError:
            # Happens when vocabulary is empty.
            return [0.0 for _ in clean_texts]

    return _compute_simple_tfidf_similarity(clean_texts, query)


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower())
        if len(token) > 2
    ]


def _compute_simple_tfidf_similarity(texts: list[str], query: str) -> list[float]:
    tokenized_docs = [_tokenize(text) for text in texts]
    query_tokens = _tokenize(query)

    if not query_tokens:
        return [0.0 for _ in texts]

    all_docs = tokenized_docs + [query_tokens]
    doc_count = len(all_docs)
    document_frequency: Counter[str] = Counter()

    for tokens in all_docs:
        document_frequency.update(set(tokens))

    def vectorize(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {
            term: (count / total) * math.log((1 + doc_count) / (1 + document_frequency[term])) + 1
            for term, count in counts.items()
        }

    query_vector = vectorize(query_tokens)
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))

    scores: list[float] = []

    for tokens in tokenized_docs:
        doc_vector = vectorize(tokens)
        doc_norm = math.sqrt(sum(value * value for value in doc_vector.values()))

        if not doc_norm or not query_norm:
            scores.append(0.0)
            continue

        dot_product = sum(
            value * query_vector.get(term, 0.0)
            for term, value in doc_vector.items()
        )
        scores.append(float(dot_product / (doc_norm * query_norm)))

    return scores


def retrieve_candidates_for_question(
    company_df: pd.DataFrame,
    question_id: str,
    top_k: int = TOP_K_CANDIDATES_PER_QUESTION,
) -> pd.DataFrame:
    """
    Retrieve candidate PA evidence for a single company and PA question.

    For Q1/Q2:
        source prior boosts official_web.

    For Q3:
        source prior boosts EDGAR/formal documents.

    Output includes:
        keyword_relevance
        context_completeness
        rag_similarity
        base_evidence_score_0_1
        source_prior
        source_adjusted_rank_score
        evidence_quality_factor
        rule_based_pa_tone_bonus
    """
    if question_id not in PA_QUESTIONS:
        raise ValueError(f"Unknown PA question_id: {question_id}")

    question = PA_QUESTIONS[question_id]

    if company_df.empty:
        return company_df.copy()

    df = company_df.copy()

    texts = df["text_clean"].fillna("").astype(str).tolist()
    rag_scores = compute_tfidf_similarity(texts, question.query)

    # Normalize section and perform section-based filtering to remove obvious non-purpose
    df["section_norm"] = df.get("section", "").fillna("").astype(str).str.lower()

    def section_is_blacklisted(section: str) -> bool:
        if not section:
            return False
        for term in SECTION_BLACKLIST:
            if term in section:
                return True
        return False

    df = df[~df["section_norm"].apply(section_is_blacklisted)].copy()

    # Recompute texts and rag_scores for filtered df
    texts = df["text_clean"].fillna("").astype(str).tolist()
    rag_scores = compute_tfidf_similarity(texts, question.query)

    df["question_id"] = question_id
    df["question_name"] = question.name
    df["question_text"] = question.question

    df["keyword_relevance"] = df["text_clean"].apply(keyword_relevance_score)

    df["context_completeness"] = df.apply(
        lambda row: context_completeness_score(
            text=row.get("text_clean", ""),
            section=row.get("section", ""),
        ),
        axis=1,
    )

    df["rag_similarity"] = rag_scores

    # PA-specific lexical flags
    norm_texts = df["text_clean"].fillna("").astype(str).str.lower()

    def contains_any(text: str, terms: list[str]) -> bool:
        if not text:
            return False
        for t in terms:
            if t in text:
                return True
        return False

    df["pa_positive_flag"] = norm_texts.apply(lambda t: float(contains_any(t, PA_POSITIVE_TERMS)))
    df["negative_boilerplate_flag"] = norm_texts.apply(lambda t: float(contains_any(t, PA_NEGATIVE_TERMS)))

    # Normalize rag similarity into 0-1 within this candidate set
    try:
        rag_series = pd.Series(df["rag_similarity"].fillna(0.0).astype(float))
        rag_min = float(rag_series.min())
        rag_max = float(rag_series.max())
        if rag_max > rag_min:
            df["rag_similarity_norm"] = (rag_series - rag_min) / (rag_max - rag_min)
        else:
            df["rag_similarity_norm"] = 0.0
    except Exception:
        df["rag_similarity_norm"] = 0.0

    # source_section_prior currently uses source_prior; keep for extensibility
    df["source_section_prior"] = df.get("source_prior", 1.0)

    # Compute pa_candidate_rank_score using configured weights and negative penalty
    kw = PA_CANDIDATE_WEIGHTS

    df["pa_candidate_rank_score"] = (
        kw.get("keyword", 0.0) * df.get("keyword_relevance", 0.0)
        + kw.get("rag_norm", 0.0) * df.get("rag_similarity_norm", 0.0)
        + kw.get("source_section_prior", 0.0) * df.get("source_section_prior", 1.0)
        + kw.get("context", 0.0) * df.get("context_completeness", 0.0)
        - NEGATIVE_BOILERPLATE_PENALTY * df.get("negative_boilerplate_flag", 0.0)
    )


    df["base_evidence_score_0_1"] = df.apply(
        lambda row: compute_base_evidence_score(
            keyword_score=row.get("keyword_relevance", 0.0),
            context_score=row.get("context_completeness", 0.0),
            rag_similarity=row.get("rag_similarity", 0.0),
        ),
        axis=1,
    )

    df["source_prior"] = df["normalized_source"].apply(
        lambda source: get_source_prior(
            question_id=question_id,
            normalized_source=source,
        )
    )

    # Source prior affects evidence selection/ranking.
    # It does not directly replace LLM rubric score.
    df["source_adjusted_rank_score"] = (
        df["base_evidence_score_0_1"] * df["source_prior"]
    )

    df["evidence_quality_factor"] = df["base_evidence_score_0_1"].apply(
        evidence_quality_factor
    )

    df["rule_based_pa_tone_bonus"] = df["text_clean"].apply(
        rule_based_pa_tone_bonus
    )

    # Sort candidates by pa_candidate_rank_score (new RAG-style ranking)
    if "pa_candidate_rank_score" in df.columns:
        df = df.sort_values(by=["pa_candidate_rank_score"], ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values(
            by=[
                "source_adjusted_rank_score",
                "base_evidence_score_0_1",
                "rag_similarity",
                "keyword_relevance",
            ],
            ascending=False,
        ).reset_index(drop=True)

    df["candidate_rank"] = range(1, len(df) + 1)

    return df.head(top_k).copy()


def select_llm_evidence_q1_q2(
    candidates_df: pd.DataFrame,
    top_k: int = TOP_K_LLM_EVIDENCE_Q1_Q2,
) -> pd.DataFrame:
    """
    Select top evidence rows for Q1/Q2 LLM scoring.

    Q1/Q2 are evidence-level tasks, so only a small number of top chunks
    should be sent to the LLM.
    """
    if candidates_df.empty:
        return candidates_df.copy()

    # Prefer the pa_candidate_rank_score if available, otherwise fall back
    if "pa_candidate_rank_score" in candidates_df.columns:
        sort_by = ["pa_candidate_rank_score", "base_evidence_score_0_1", "rag_similarity"]
    else:
        sort_by = ["source_adjusted_rank_score", "base_evidence_score_0_1", "rag_similarity"]

    selected = (
        candidates_df.sort_values(by=sort_by, ascending=False)
        .head(top_k)
        .copy()
        .reset_index(drop=True)
    )

    selected["llm_input_rank"] = range(1, len(selected) + 1)

    return selected


def build_q3_evidence_set(
    company_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build evidence set for Q3: Distinction from Branding.

    Q3 should not be scored as isolated evidence only.
    It needs a source / document-type comparison:
        - official_web
        - edgar
        - linkedin, if available

    This function therefore selects:
        1. Top evidence per source
        2. Top global evidence
        3. Deduplicated combined evidence set
    """
    q3_candidates = retrieve_candidates_for_question(
        company_df=company_df,
        question_id="PA_Q3",
        top_k=max(TOP_K_CANDIDATES_PER_QUESTION, 40),
    )

    if q3_candidates.empty:
        return q3_candidates.copy()

    selected_parts: list[pd.DataFrame] = []

    for _, group in q3_candidates.groupby("normalized_source"):
        top_by_source = (
            group.sort_values(
                by=[
                    "source_adjusted_rank_score",
                    "base_evidence_score_0_1",
                    "rag_similarity",
                ],
                ascending=False,
            )
            .head(TOP_K_EVIDENCE_PER_SOURCE_Q3)
            .copy()
        )

        selected_parts.append(top_by_source)

    source_balanced = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else pd.DataFrame()
    )

    global_top = q3_candidates.head(TOP_K_GLOBAL_EVIDENCE_Q3).copy()

    combined = pd.concat(
        [source_balanced, global_top],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["chunk_id"],
        keep="first",
    ).copy()

    combined = combined.sort_values(
        by=[
            "normalized_source",
            "source_adjusted_rank_score",
            "base_evidence_score_0_1",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    combined["evidence_set_rank"] = range(1, len(combined) + 1)

    return combined


def retrieve_all_pa_candidates_for_company(
    company_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Convenience function for diagnostics/debug.

    Returns candidate evidence for all PA questions.
    """
    result = {}

    for question_id in PA_QUESTIONS:
        if question_id == "PA_Q3":
            result[question_id] = build_q3_evidence_set(company_df)
        else:
            candidates = retrieve_candidates_for_question(company_df, question_id)
            result[question_id] = select_llm_evidence_q1_q2(candidates)

    return result
