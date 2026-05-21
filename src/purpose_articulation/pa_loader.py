from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from purpose_articulation.pa_config import COLUMN_ALIASES, SOURCE_ALIASES


STANDARD_COLUMNS = [
    "company",
    "year",
    "source",
    "normalized_source",
    "section",
    "subsection",
    "chunk_id",
    "doc_id",
    "text_clean",
    "text_raw",
]


def _pick_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    """
    Find the actual input column corresponding to a canonical column name.

    Example:
        canonical_name = "text_clean"
        acceptable input columns = ["text_clean", "clean_text", "text", "chunk_text"]
    """
    candidates = COLUMN_ALIASES.get(canonical_name, [canonical_name])
    lower_map = {str(col).lower().strip(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lower_map:
            return lower_map[key]

    return None


def normalize_source_name(source: str | None) -> str:
    """
    Normalize raw source labels into:
        official_web
        edgar
        linkedin
        unknown
    """
    if source is None or pd.isna(source):
        return "unknown"

    s = str(source).strip().lower()

    if not s:
        return "unknown"

    for normalized_source, aliases in SOURCE_ALIASES.items():
        for alias in aliases:
            alias_norm = alias.lower().strip()
            if alias_norm and alias_norm in s:
                return normalized_source

    return s


def _make_stable_chunk_id(row: pd.Series, idx: int) -> str:
    """
    Create a stable fallback chunk_id if the input file lacks one.
    """
    base = "|".join(
        [
            str(row.get("company", "")),
            str(row.get("year", "")),
            str(row.get("source", "")),
            str(row.get("section", "")),
            str(row.get("text_clean", ""))[:300],
        ]
    )

    digest = hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"auto_chunk_{idx}_{digest}"


def _coerce_year(series: pd.Series) -> pd.Series:
    """
    Convert year-like values to nullable integer.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _standardize_input_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw input columns into the standard PA pipeline schema.
    """
    mapped = pd.DataFrame()

    for canonical_col in STANDARD_COLUMNS:
        if canonical_col == "normalized_source":
            continue

        raw_col = _pick_column(df_raw, canonical_col)

        if raw_col is not None:
            mapped[canonical_col] = df_raw[raw_col]
        else:
            mapped[canonical_col] = None

    return mapped


def validate_loaded_chunks(df: pd.DataFrame) -> None:
    """
    Basic validation after loading and standardization.
    """
    required = ["company", "source", "normalized_source", "text_clean", "chunk_id"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required standardized columns: {missing}")

    if df.empty:
        raise ValueError("Loaded chunk dataframe is empty after filtering.")

    if df["company"].eq("").all():
        raise ValueError("All company values are empty.")

    if df["text_clean"].str.len().max() < 20:
        raise ValueError("No meaningful text found in text_clean.")


def load_chunks(input_path: str | Path) -> pd.DataFrame:
    """
    Load unified_chunks_final_v4.csv and standardize it for PA scoring.

    Output schema:
        company
        year
        source
        normalized_source
        section
        subsection
        chunk_id
        doc_id
        text_clean
        text_raw
        input_row_id
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input chunks file not found: {input_path}")

    df_raw = pd.read_csv(input_path, low_memory=False)
    df = _standardize_input_columns(df_raw)

    df["input_row_id"] = range(len(df))

    # -------------------------------------------------------------------------
    # Text fallback
    # -------------------------------------------------------------------------
    df["text_clean"] = df["text_clean"].fillna("").astype(str).str.strip()
    df["text_raw"] = df["text_raw"].fillna("").astype(str).str.strip()

    # If text_clean is empty but text_raw exists, use text_raw.
    clean_empty = df["text_clean"].eq("")
    df.loc[clean_empty, "text_clean"] = df.loc[clean_empty, "text_raw"]

    # If text_raw is empty, keep a copy of text_clean.
    raw_empty = df["text_raw"].eq("")
    df.loc[raw_empty, "text_raw"] = df.loc[raw_empty, "text_clean"]

    # -------------------------------------------------------------------------
    # Basic field cleanup
    # -------------------------------------------------------------------------
    df["company"] = df["company"].fillna("").astype(str).str.strip()
    df["source"] = df["source"].fillna("unknown").astype(str).str.strip()
    df["normalized_source"] = df["source"].apply(normalize_source_name)

    df["section"] = df["section"].fillna("").astype(str).str.strip()
    df["subsection"] = df["subsection"].fillna("").astype(str).str.strip()
    df["doc_id"] = df["doc_id"].fillna("").astype(str).str.strip()

    df["year"] = _coerce_year(df["year"])

    # -------------------------------------------------------------------------
    # chunk_id handling
    # -------------------------------------------------------------------------
    df["chunk_id"] = df["chunk_id"].fillna("").astype(str).str.strip()

    missing_chunk_id = df["chunk_id"].eq("")
    if missing_chunk_id.any():
        for idx in df.index[missing_chunk_id]:
            df.loc[idx, "chunk_id"] = _make_stable_chunk_id(df.loc[idx], idx)

    # -------------------------------------------------------------------------
    # Filter unusable rows
    # -------------------------------------------------------------------------
    df = df[
        (df["company"] != "")
        & (df["text_clean"].str.len() >= 20)
    ].copy()

    # Avoid duplicate chunk rows inside the same company.
    df = df.drop_duplicates(
        subset=["company", "chunk_id"],
        keep="first",
    ).reset_index(drop=True)

    validate_loaded_chunks(df)

    return df


def build_company_targets(df: pd.DataFrame) -> list[str]:
    """
    Build company-level targets.

    PA pilot no longer scores company-year targets.
    All years are pooled for each company.
    """
    companies = (
        df["company"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    companies = sorted([c for c in companies.unique().tolist() if c])

    return companies


def filter_company(df: pd.DataFrame, company: str) -> pd.DataFrame:
    """
    Return all pooled-year evidence for a single company.
    """
    company_norm = str(company).strip()

    return df[df["company"].astype(str).str.strip() == company_norm].copy()


def summarize_loaded_chunks(df: pd.DataFrame) -> dict:
    """
    Produce quick diagnostics for terminal/report output.
    """
    year_valid = pd.to_numeric(df["year"], errors="coerce").dropna()

    return {
        "total_chunks": int(len(df)),
        "total_companies": int(df["company"].nunique()),
        "source_counts": df["normalized_source"].value_counts(dropna=False).to_dict(),
        "year_min": int(year_valid.min()) if not year_valid.empty else None,
        "year_max": int(year_valid.max()) if not year_valid.empty else None,
        "missing_year_count": int(df["year"].isna().sum()),
        "empty_section_count": int(df["section"].eq("").sum()),
    }
