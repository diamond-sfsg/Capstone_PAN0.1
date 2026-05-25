from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def coerce_year_column(df: pd.DataFrame, column: str = "year") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out


def drop_missing_years(df: pd.DataFrame, column: str = "year") -> pd.DataFrame:
    out = coerce_year_column(df, column=column)
    return out[out[column].notna()].copy().reset_index(drop=True)


def build_company_year_targets(
    df: pd.DataFrame,
    company_filter: Iterable[str] | None = None,
    year_filter: Iterable[int] | None = None,
) -> pd.DataFrame:
    if "company" not in df.columns or "year" not in df.columns:
        raise ValueError("Expected columns: company, year")

    work = drop_missing_years(df)
    work["company"] = work["company"].fillna("").astype(str).str.strip()
    work = work[work["company"].ne("")].copy()

    if company_filter:
        companies = {str(company).strip() for company in company_filter}
        work = work[work["company"].isin(companies)].copy()

    if year_filter:
        years = {int(year) for year in year_filter}
        work = work[work["year"].astype(int).isin(years)].copy()

    targets = (
        work.groupby(["company", "year"], dropna=False)
        .size()
        .reset_index(name="chunk_count")
        .sort_values(["company", "year"])
        .reset_index(drop=True)
    )
    targets["year"] = targets["year"].astype(int)
    return targets


def attach_target_year(records: list[dict], target_year: int) -> list[dict]:
    out: list[dict] = []
    for record in records:
        item = dict(record)
        item["target_year"] = int(target_year)
        out.append(item)
    return out

