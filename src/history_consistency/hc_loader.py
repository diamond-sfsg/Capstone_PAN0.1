# src/history_consistency/hc_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from history_consistency.hc_config import (
    INPUT_CHUNKS_CSV,
    REQUIRED_INPUT_COLUMNS,
    OPTIONAL_INPUT_COLUMNS,
    YEAR_WINDOW,
    MIN_EVIDENCE_PER_COMPANY,
)


@dataclass(frozen=True)
class CompanyEvidencePool:
    """
    Company-level evidence pool for HC scoring.

    HC is company-level, not company-year-level.
    For each company, we keep up to the latest 10-year window available
    in the input corpus.
    """

    company: str
    latest_year: Optional[int]
    start_year: Optional[int]
    end_year: Optional[int]
    evidence_count: int
    distinct_year_count: int
    data: pd.DataFrame


def load_chunks(input_path: Path | str = INPUT_CHUNKS_CSV) -> pd.DataFrame:
    """
    Load unified chunk corpus.

    Expected main input:
    data/clean_2.0/unified_chunks_final_v4.csv
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input chunk file not found: {path}")

    df = pd.read_csv(path)
    validate_input_columns(df)
    df = standardize_chunks(df)

    return df


def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Validate required columns exist.
    """
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "Input chunk file is missing required columns: "
            + ", ".join(missing)
        )


def standardize_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize input dataframe types and basic values.

    This function does not perform aggressive filtering.
    It keeps the corpus as intact as possible and only removes rows that
    cannot be scored at all.
    """
    out = df.copy()

    # Ensure optional columns exist, so downstream modules do not break.
    for col in OPTIONAL_INPUT_COLUMNS:
        if col not in out.columns:
            out[col] = None

    # Standardize required fields.
    out["chunk_id"] = out["chunk_id"].astype(str)
    out["company"] = out["company"].astype(str).str.strip()
    out["source"] = out["source"].fillna("").astype(str).str.lower().str.strip()
    out["section"] = out["section"].fillna("").astype(str).str.lower().str.strip()
    out["text_clean"] = out["text_clean"].fillna("").astype(str)

    # Year handling.
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["company", "text_clean"])
    out = out[out["company"].str.len() > 0]
    out = out[out["text_clean"].str.len() > 0]

    # Keep rows with missing year, but they will not contribute to year coverage.
    # For actual time-window filtering, valid years are used.
    out["year"] = out["year"].astype("Int64")

    # Optional string columns.
    for col in ["doc_id", "source_file", "subsection", "quality_flag"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    return out.reset_index(drop=True)


def filter_quality_rows(
    df: pd.DataFrame,
    allowed_quality_flags: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Optional quality filter.

    By default, this keeps all rows because HC may need historical evidence,
    and aggressive filtering can remove useful continuity signals.

    If allowed_quality_flags is provided, only those quality flags are kept.
    """
    if allowed_quality_flags is None:
        return df.copy()

    allowed = {x.lower().strip() for x in allowed_quality_flags}

    if "quality_flag" not in df.columns:
        return df.copy()

    out = df.copy()
    out["quality_flag"] = out["quality_flag"].fillna("").astype(str).str.lower()
    out = out[out["quality_flag"].isin(allowed)]

    return out.reset_index(drop=True)


def build_company_evidence_pools(
    df: pd.DataFrame,
    year_window: int = YEAR_WINDOW,
    min_evidence_per_company: int = MIN_EVIDENCE_PER_COMPANY,
) -> Dict[str, CompanyEvidencePool]:
    """
    Build company-level evidence pools.

    Logic:
    - HC is company-level.
    - For each company, identify its latest available valid year.
    - Keep evidence from latest_year - year_window + 1 to latest_year.
    - Rows without valid year are retained only if the company has no valid years.
    """
    pools: Dict[str, CompanyEvidencePool] = {}

    for company, group in df.groupby("company", dropna=False):
        company_name = str(company).strip()
        if not company_name:
            continue

        group = group.copy()

        valid_years = (
            group["year"]
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )

        if valid_years:
            latest_year = max(valid_years)
            start_year = latest_year - year_window + 1
            end_year = latest_year

            pool_df = group[
                group["year"].isna()
                | (
                    (group["year"].astype("Int64") >= start_year)
                    & (group["year"].astype("Int64") <= end_year)
                )
            ].copy()

            distinct_year_count = (
                pool_df["year"].dropna().astype(int).nunique()
            )
        else:
            latest_year = None
            start_year = None
            end_year = None
            pool_df = group.copy()
            distinct_year_count = 0

        evidence_count = len(pool_df)

        if evidence_count < min_evidence_per_company:
            # Keep it in the pool anyway.
            # Later modules will mark needs_human_review.
            pass

        pools[company_name] = CompanyEvidencePool(
            company=company_name,
            latest_year=latest_year,
            start_year=start_year,
            end_year=end_year,
            evidence_count=evidence_count,
            distinct_year_count=distinct_year_count,
            data=pool_df.reset_index(drop=True),
        )

    return pools


def summarize_company_pools(
    pools: Dict[str, CompanyEvidencePool],
) -> pd.DataFrame:
    """
    Return a compact diagnostics table for company evidence pools.
    """
    rows: List[dict] = []

    for company, pool in pools.items():
        rows.append(
            {
                "company": company,
                "latest_year": pool.latest_year,
                "start_year": pool.start_year,
                "end_year": pool.end_year,
                "evidence_count": pool.evidence_count,
                "distinct_year_count": pool.distinct_year_count,
                "needs_human_review": pool.evidence_count < MIN_EVIDENCE_PER_COMPANY,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["needs_human_review", "company"],
        ascending=[False, True],
    )


def load_company_evidence_pools(
    input_path: Path | str = INPUT_CHUNKS_CSV,
) -> Dict[str, CompanyEvidencePool]:
    """
    Convenience wrapper:
    load chunks -> standardize -> build company evidence pools.
    """
    df = load_chunks(input_path)
    return build_company_evidence_pools(df)


if __name__ == "__main__":
    chunks = load_chunks()
    pools = build_company_evidence_pools(chunks)
    summary = summarize_company_pools(pools)

    print("Loaded chunks:", len(chunks))
    print("Company pools:", len(pools))
    print(summary.head(20).to_string(index=False))
