from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "chunk_id",
    "company",
    "year",
    "source",
    "section",
    "subsection",
    "text_clean",
]

OPTIONAL_COLUMNS_WITH_DEFAULTS = {
    "text_raw": "",
    "duplicate_group": None,
    "similarity_scope": "none",
}


def load_chunk_dataframe(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in input CSV: {missing}"
        )

    df = df.copy()
    for col, default in OPTIONAL_COLUMNS_WITH_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    df["chunk_id"] = df["chunk_id"].astype(str)
    duplicate_mask = df["chunk_id"].duplicated(keep=False)
    if duplicate_mask.any():
        df["original_chunk_id"] = df["chunk_id"]
        occurrence = df.groupby("chunk_id").cumcount().astype(str).str.zfill(4)
        df.loc[duplicate_mask, "chunk_id"] = (
            df.loc[duplicate_mask, "chunk_id"]
            + "|dup"
            + occurrence.loc[duplicate_mask]
        )

    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)

    return df


def load_chunk_corpus(csv_path: str | Path) -> pd.DataFrame:
    return load_chunk_dataframe(csv_path)
