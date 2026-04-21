from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import INPUT_CSV, REQUIRED_COLUMNS

## 读取和字段检查，没有scoring


def load_chunk_corpus(csv_path: Path | str = INPUT_CSV) -> pd.DataFrame:
    """
    Load the normalized chunk corpus from CSV and validate required columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the unified chunk corpus.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing)
        )

    df = _standardize_columns(df)
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize basic field formats without changing semantic content.
    """
    out = df.copy()

    text_cols = ["text_raw", "text_clean", "section", "subsection", "source", "source_file", "quality_flag"]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    if "company" in out.columns:
        out["company"] = out["company"].fillna("").astype(str)

    if "doc_id" in out.columns:
        out["doc_id"] = out["doc_id"].fillna("").astype(str)

    if "chunk_id" in out.columns:
        out["chunk_id"] = out["chunk_id"].fillna("").astype(str)

    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")

    if "token_count" in out.columns:
        out["token_count"] = pd.to_numeric(out["token_count"], errors="coerce").fillna(0).astype(int)

    if "char_count" in out.columns:
        out["char_count"] = pd.to_numeric(out["char_count"], errors="coerce").fillna(0).astype(int)

    if "is_short_text" in out.columns:
        out["is_short_text"] = out["is_short_text"].fillna("").astype(str)

    if "is_duplicate_like" in out.columns:
        out["is_duplicate_like"] = out["is_duplicate_like"].fillna("").astype(str)

    if "duplicate_group" in out.columns:
        out["duplicate_group"] = out["duplicate_group"].fillna("").astype(str)

    return out


def select_base_columns(df: pd.DataFrame, extra_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Select the standard retrieval columns plus any optional extra columns.
    """
    base_cols = [
        "chunk_id",
        "doc_id",
        "company",
        "year",
        "source",
        "source_file",
        "section",
        "subsection",
        "text_raw",
        "text_clean",
        "token_count",
        "char_count",
        "is_short_text",
        "is_duplicate_like",
        "duplicate_group",
        "quality_flag",
        "normalize_version",
    ]

    if extra_columns:
        for col in extra_columns:
            if col not in base_cols:
                base_cols.append(col)

    keep_cols = [c for c in base_cols if c in df.columns]
    return df[keep_cols].copy()