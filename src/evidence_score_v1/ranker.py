from __future__ import annotations

import pandas as pd


def add_score_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    score_cols = [
        "lexical_score",
        "tfidf_score",
        "embedding_score",
        "metadata_score",
    ]

    for col in score_cols:
        rank_col = col.replace("_score", "_rank")
        if col in out.columns:
            out[rank_col] = out[col].rank(method="dense", ascending=False).astype(int)

    return out