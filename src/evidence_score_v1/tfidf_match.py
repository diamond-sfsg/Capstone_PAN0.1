from __future__ import annotations

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    TfidfVectorizer = None
    cosine_similarity = None

from .config import DEFAULT_TFIDF_CONFIG, TfidfConfig


def _word_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    tokens = str(text).lower().split()
    grams: list[str] = []
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        grams.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return grams


def _fallback_query_cosine_scores(
    texts: list[str],
    query_text: str,
    ngram_range: tuple[int, int],
) -> list[float]:
    query_terms = set(_word_ngrams(query_text, ngram_range))
    if not query_terms:
        return [0.0 for _ in texts]

    query_norm = len(query_terms) ** 0.5
    scores = []
    for text in texts:
        terms = _word_ngrams(text, ngram_range)
        if not terms:
            scores.append(0.0)
            continue
        hits = sum(1 for term in terms if term in query_terms)
        scores.append(float(hits / ((len(terms) ** 0.5) * query_norm)))
    return scores


def score_tfidf(
    df: pd.DataFrame,
    cfg,
    tfidf_cfg: TfidfConfig = DEFAULT_TFIDF_CONFIG,
) -> pd.DataFrame:
    texts = df["text_for_match"].tolist()
    doc_count = len(texts)
    if doc_count == 0:
        return pd.DataFrame({"tfidf_cosine": []})

    if TfidfVectorizer is None:
        sims = _fallback_query_cosine_scores(
            texts,
            query_text=cfg.query_text,
            ngram_range=tfidf_cfg.ngram_range,
        )
        out = pd.DataFrame({"tfidf_cosine": sims})
        out["tfidf_rank_within_dimension"] = out["tfidf_cosine"].rank(method="first", ascending=False).astype(int)
        return out

    min_df = min(tfidf_cfg.min_df, doc_count)
    max_df = tfidf_cfg.max_df

    if isinstance(max_df, float) and max_df * doc_count < min_df:
        min_df = max(1, int(max_df * doc_count))
        if max_df * doc_count < min_df:
            max_df = 1.0

    vectorizer = TfidfVectorizer(
        lowercase=tfidf_cfg.lowercase,
        stop_words=tfidf_cfg.stop_words,
        ngram_range=tfidf_cfg.ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=tfidf_cfg.sublinear_tf,
        max_features=tfidf_cfg.max_features,
    )
    X = vectorizer.fit_transform(texts)
    q = vectorizer.transform([cfg.query_text])
    sims = cosine_similarity(X, q).ravel()
    out = pd.DataFrame({"tfidf_cosine": sims})
    out["tfidf_rank_within_dimension"] = out["tfidf_cosine"].rank(method="first", ascending=False).astype(int)
    return out


def compute_tfidf_scores(
    df: pd.DataFrame,
    cfg,
    tfidf_cfg: TfidfConfig = DEFAULT_TFIDF_CONFIG,
) -> pd.DataFrame:
    out = score_tfidf(df, cfg, tfidf_cfg=tfidf_cfg)
    out.insert(0, "chunk_id", df["chunk_id"].to_numpy())
    out["tfidf_score"] = out["tfidf_cosine"]
    return out
