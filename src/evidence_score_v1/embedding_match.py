from __future__ import annotations

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    TfidfVectorizer = None
    cosine_similarity = None


def _char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    padded = f" {str(text).lower()} "
    grams: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(padded) < n:
            continue
        grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


def _fallback_char_cosine_scores(texts: list[str], query_text: str) -> list[float]:
    query_terms = set(_char_ngrams(query_text))
    if not query_terms:
        return [0.0 for _ in texts]

    query_norm = len(query_terms) ** 0.5
    scores = []
    for text in texts:
        normalized = str(text).lower()
        if not normalized:
            scores.append(0.0)
            continue
        hits = sum(1 for term in query_terms if term in normalized)
        doc_norm = max(len(normalized), 1) ** 0.5
        scores.append(float(hits / (doc_norm * query_norm)))
    return scores


def score_embedding_proxy(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """A lightweight dense-retrieval proxy.

    This uses char-level TF-IDF as a placeholder so the pipeline runs without
    external embedding dependencies. Replace with real embeddings later.
    """
    if df.empty:
        return pd.DataFrame({"embedding_cosine": []})

    if TfidfVectorizer is None:
        sims = _fallback_char_cosine_scores(df["text_for_match"].tolist(), cfg.query_text)
        out = pd.DataFrame({"embedding_cosine": sims})
        out["embedding_rank_within_dimension"] = out["embedding_cosine"].rank(method="first", ascending=False).astype(int)
        return out

    # Limit features to prevent memory issues with large datasets
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", 
        ngram_range=(3, 5), 
        min_df=1,
        max_features=30000  # Limit to prevent excessive memory usage
    )
    X = vectorizer.fit_transform(df["text_for_match"].tolist())
    q = vectorizer.transform([cfg.query_text])
    sims = cosine_similarity(X, q).ravel()
    out = pd.DataFrame({"embedding_cosine": sims})
    out["embedding_rank_within_dimension"] = out["embedding_cosine"].rank(method="first", ascending=False).astype(int)
    return out


def compute_embedding_scores(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = score_embedding_proxy(df, cfg)
    out.insert(0, "chunk_id", df["chunk_id"].to_numpy())
    out["embedding_score"] = out["embedding_cosine"]
    return out
