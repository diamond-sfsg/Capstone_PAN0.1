# src/history_consistency/hc_bonus.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from history_consistency.hc_config import (
    HC_HISTORY_BONUS_ENABLED,
    HC_HISTORY_BONUS_MAX,
    HC_HISTORY_BONUS_WEIGHTS,
    HISTORICAL_RECENT_SPLIT_YEARS,
    DUPLICATE_SCOPE_COLUMN,
    DUPLICATE_GROUP_COLUMN,
    CROSS_YEAR_RECURRING_VALUES,
)


@dataclass(frozen=True)
class HCHistoryBonusResult:
    """
    HC history bonus result.

    Bonus is evidence-set-level, not single-evidence-level.
    """

    bonus: float
    distinct_year_count: int
    has_multi_year_coverage: bool
    has_three_plus_year_coverage: bool
    has_cross_year_recurring_theme: bool
    has_historical_and_recent_coverage: bool


def normalize_value(value: object) -> str:
    """
    Normalize value for matching.
    """
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def get_distinct_year_count(df: pd.DataFrame) -> int:
    """
    Count distinct valid years in evidence set.
    """
    if "year" not in df.columns or df.empty:
        return 0

    years = pd.to_numeric(df["year"], errors="coerce").dropna()

    if years.empty:
        return 0

    return int(years.astype(int).nunique())


def has_cross_year_recurring_scope(df: pd.DataFrame) -> bool:
    """
    Detect cross-year recurrence using similarity_scope.
    """
    if DUPLICATE_SCOPE_COLUMN not in df.columns or df.empty:
        return False

    scopes = df[DUPLICATE_SCOPE_COLUMN].apply(normalize_value)

    for scope in scopes:
        if scope in CROSS_YEAR_RECURRING_VALUES:
            return True
        if "cross_year" in scope or "recurring" in scope:
            return True

    return False


def has_cross_year_duplicate_group(df: pd.DataFrame) -> bool:
    """
    Detect whether any duplicate_group appears across multiple years.
    """
    if (
        DUPLICATE_GROUP_COLUMN not in df.columns
        or "year" not in df.columns
        or df.empty
    ):
        return False

    temp = df.copy()
    temp[DUPLICATE_GROUP_COLUMN] = temp[DUPLICATE_GROUP_COLUMN].apply(normalize_value)
    temp["year_numeric"] = pd.to_numeric(temp["year"], errors="coerce")

    temp = temp[
        (temp[DUPLICATE_GROUP_COLUMN] != "")
        & temp["year_numeric"].notna()
    ]

    if temp.empty:
        return False

    year_counts = temp.groupby(DUPLICATE_GROUP_COLUMN)["year_numeric"].nunique()

    return bool((year_counts >= 2).any())


def has_cross_year_recurring_theme(df: pd.DataFrame) -> bool:
    """
    Detect whether evidence set has recurring purpose theme across years.

    Uses:
    - similarity_scope if available
    - duplicate_group repeated across years if available
    - hc_redundancy_type == cross_year_recurring if available
    """
    if df.empty:
        return False

    if "hc_redundancy_type" in df.columns:
        values = df["hc_redundancy_type"].apply(normalize_value)
        if (values == "cross_year_recurring").any():
            return True

    if has_cross_year_recurring_scope(df):
        return True

    if has_cross_year_duplicate_group(df):
        return True

    return False


def has_historical_and_recent_coverage(
    df: pd.DataFrame,
    split_years: int = HISTORICAL_RECENT_SPLIT_YEARS,
) -> bool:
    """
    Check whether the evidence set includes both older and recent evidence.

    Definition:
    - latest_year = max(valid years)
    - recent evidence: year > latest_year - split_years
    - historical evidence: year <= latest_year - split_years
    """
    if "year" not in df.columns or df.empty:
        return False

    years = pd.to_numeric(df["year"], errors="coerce").dropna()

    if years.empty:
        return False

    years = years.astype(int)
    latest_year = int(years.max())
    cutoff = latest_year - split_years

    has_recent = bool((years > cutoff).any())
    has_historical = bool((years <= cutoff).any())

    return has_recent and has_historical


def compute_hc_history_bonus(
    df: pd.DataFrame,
) -> HCHistoryBonusResult:
    """
    Compute HC history bonus.

    Bonus logic:
    - 2+ years coverage: +0.15
    - 3+ years coverage: +0.15
    - cross-year recurring theme: +0.15
    - both historical and recent coverage: +0.05

    Max:
    0.50
    """
    if not HC_HISTORY_BONUS_ENABLED or df.empty:
        return HCHistoryBonusResult(
            bonus=0.0,
            distinct_year_count=0,
            has_multi_year_coverage=False,
            has_three_plus_year_coverage=False,
            has_cross_year_recurring_theme=False,
            has_historical_and_recent_coverage=False,
        )

    distinct_year_count = get_distinct_year_count(df)

    has_multi_year = distinct_year_count >= 2
    has_three_plus = distinct_year_count >= 3
    has_recurring = has_cross_year_recurring_theme(df)
    has_old_recent = has_historical_and_recent_coverage(df)

    bonus = 0.0

    if has_multi_year:
        bonus += HC_HISTORY_BONUS_WEIGHTS["multi_year_coverage"]

    if has_three_plus:
        bonus += HC_HISTORY_BONUS_WEIGHTS["three_plus_year_coverage"]

    if has_recurring:
        bonus += HC_HISTORY_BONUS_WEIGHTS["cross_year_recurring_theme"]

    if has_old_recent:
        bonus += HC_HISTORY_BONUS_WEIGHTS["historical_and_recent_coverage"]

    bonus = min(float(bonus), float(HC_HISTORY_BONUS_MAX))

    return HCHistoryBonusResult(
        bonus=bonus,
        distinct_year_count=distinct_year_count,
        has_multi_year_coverage=has_multi_year,
        has_three_plus_year_coverage=has_three_plus,
        has_cross_year_recurring_theme=has_recurring,
        has_historical_and_recent_coverage=has_old_recent,
    )


def bonus_result_to_dict(result: HCHistoryBonusResult) -> Dict[str, object]:
    """
    Convert bonus result dataclass to dict.
    """
    return {
        "hc_history_bonus": result.bonus,
        "distinct_year_count": result.distinct_year_count,
        "has_multi_year_coverage": result.has_multi_year_coverage,
        "has_three_plus_year_coverage": result.has_three_plus_year_coverage,
        "has_cross_year_recurring_theme": result.has_cross_year_recurring_theme,
        "has_historical_and_recent_coverage": result.has_historical_and_recent_coverage,
    }


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "chunk_id": ["a", "b", "c"],
            "year": [2018, 2021, 2024],
            "duplicate_group": ["g1", "g1", ""],
            "similarity_scope": ["cross_year_recurring", "cross_year_recurring", ""],
        }
    )

    result = compute_hc_history_bonus(sample)
    print(bonus_result_to_dict(result))
