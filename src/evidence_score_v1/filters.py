from __future__ import annotations

import pandas as pd

from .config import (
    DROP_QUALITY_FLAGS,
    DROP_TOO_LONG,
    MAX_TOKEN_COUNT,
    MIN_TOKEN_COUNT,
    TOO_LONG_FLAG,
)

#过滤和标记，retrival satge only


def add_retrieval_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add retrieval_status for Phase 2 evidence retrieval.

    Status values:
    - keep
    - drop_garbled
    - drop_too_long
    - drop_too_short
    """
    out = df.copy()

    out["retrieval_status"] = out.apply(_determine_status, axis=1)
    return out


def filter_retrieval_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows kept for evidence retrieval.
    """
    if "retrieval_status" not in df.columns:
        df = add_retrieval_status(df)

    return df[df["retrieval_status"] == "keep"].copy()


def summarize_filter_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize retrieval_status counts for reporting.
    """
    if "retrieval_status" not in df.columns:
        df = add_retrieval_status(df)

    summary = (
        df["retrieval_status"]
        .value_counts(dropna=False)
        .rename_axis("retrieval_status")
        .reset_index(name="count")
    )
    return summary


def _determine_status(row: pd.Series) -> str:
    quality_flag = str(row.get("quality_flag", "")).strip().lower()
    token_count = int(row.get("token_count", 0) or 0)

    if quality_flag in {q.lower() for q in DROP_QUALITY_FLAGS}:
        return "drop_garbled"

    if DROP_TOO_LONG:
        if quality_flag == TOO_LONG_FLAG.lower():
            return "drop_too_long"
        if token_count > MAX_TOKEN_COUNT:
            return "drop_too_long"

    if token_count < MIN_TOKEN_COUNT:
        return "drop_too_short"

    return "keep"