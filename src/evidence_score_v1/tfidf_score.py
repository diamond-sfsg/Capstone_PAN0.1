from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_scores(
    df: pd.DataFrame,
    query_text: str,
    text_col: str = "text_clean",
) -> pd.DataFrame:
    out = df.copy()

    texts = out[text_col].fillna("").astype(str).tolist()
    if not texts:
        out["tfidf_score"] = []
        return out

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )

    x_chunks = vectorizer.fit_transform(texts)
    x_query = vectorizer.transform([query_text])

    scores = cosine_similarity(x_chunks, x_query).ravel()
    out["tfidf_score"] = scores
    return out