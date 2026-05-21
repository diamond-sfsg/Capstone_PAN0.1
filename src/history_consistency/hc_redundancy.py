# src/history_consistency/hc_redundancy.py

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from history_consistency.hc_config import (
    HC_REDUNDANCY_FACTOR,
    DUPLICATE_SCOPE_COLUMN,
    DUPLICATE_GROUP_COLUMN,
    CROSS_YEAR_RECURRING_VALUES,
    SAME_YEAR_DUPLICATE_VALUES,
)


def normalize_duplicate_value(value: object) -> str:
    """
    Normalize duplicate / similarity scope value.
    """
    if value is None or pd.isna(value):
        return ""

    return str(value).strip().lower()


def classify_redundancy_from_similarity_scope(scope: object) -> Optional[str]:
    """
    Classify redundancy type directly from similarity_scope if available.
    """
    scope_norm = normalize_duplicate_value(scope)

    if not scope_norm:
        return None

    if scope_norm in CROSS_YEAR_RECURRING_VALUES:
        return "cross_year_recurring"

    if scope_norm in SAME_YEAR_DUPLICATE_VALUES:
        if "exact" in scope_norm:
            return "same_year_exact_duplicate"
        return "same_year_near_duplicate"

    if "cross_year" in scope_norm or "recurring" in scope_norm:
        return "cross_year_recurring"

    if "exact" in scope_norm and "same" in scope_norm:
        return "same_year_exact_duplicate"

    if "near" in scope_norm and "same" in scope_norm:
        return "same_year_near_duplicate"

    if "same_year" in scope_norm:
        return "same_year_near_duplicate"

    return None


def infer_redundancy_type_for_group(group: pd.DataFrame) -> str:
    """
    Infer redundancy type for a duplicate group.

    HC-specific rule:
    - Cross-year recurrence is not penalized.
    - Same-year duplication is penalized.
    - If no year information exists, treat as same-doc repeated boilerplate
      only when there are multiple rows from the same doc/source_file.
    """
    if group.empty:
        return "unique_evidence"

    # Use explicit similarity_scope first.
    if DUPLICATE_SCOPE_COLUMN in group.columns:
        explicit_types = [
            classify_redundancy_from_similarity_scope(v)
            for v in group[DUPLICATE_SCOPE_COLUMN].tolist()
        ]
        explicit_types = [x for x in explicit_types if x is not None]

        if "cross_year_recurring" in explicit_types:
            return "cross_year_recurring"

        if "same_year_exact_duplicate" in explicit_types:
            return "same_year_exact_duplicate"

        if "same_year_near_duplicate" in explicit_types:
            return "same_year_near_duplicate"

    # Infer from year distribution.
    if "year" in group.columns:
        valid_years = group["year"].dropna()

        if not valid_years.empty:
            distinct_years = valid_years.astype(int).nunique()

            if distinct_years >= 2:
                return "cross_year_recurring"

            if len(group) >= 2 and distinct_years == 1:
                return "same_year_near_duplicate"

    # Infer repeated boilerplate from doc_id / source_file.
    for col in ["doc_id", "source_file"]:
        if col in group.columns:
            non_empty = group[col].dropna().astype(str).str.strip()
            non_empty = non_empty[non_empty != ""]

            if not non_empty.empty and non_empty.nunique() == 1 and len(group) >= 2:
                return "same_doc_repeated_boilerplate"

    if len(group) >= 2:
        return "same_year_near_duplicate"

    return "unique_evidence"


def classify_row_redundancy(
    row: pd.Series,
    group_redundancy_map: Dict[str, str],
) -> str:
    """
    Classify redundancy type for one row.

    Priority:
    1. similarity_scope
    2. duplicate_group-level inference
    3. unique_evidence
    """
    if DUPLICATE_SCOPE_COLUMN in row.index:
        explicit = classify_redundancy_from_similarity_scope(
            row.get(DUPLICATE_SCOPE_COLUMN)
        )
        if explicit is not None:
            return explicit

    if DUPLICATE_GROUP_COLUMN in row.index:
        group_value = row.get(DUPLICATE_GROUP_COLUMN)
        group_key = normalize_duplicate_value(group_value)

        if group_key and group_key in group_redundancy_map:
            return group_redundancy_map[group_key]

    return "unique_evidence"


def build_duplicate_group_redundancy_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build duplicate_group -> redundancy_type mapping.
    """
    if DUPLICATE_GROUP_COLUMN not in df.columns:
        return {}

    out: Dict[str, str] = {}

    temp = df.copy()
    temp[DUPLICATE_GROUP_COLUMN] = (
        temp[DUPLICATE_GROUP_COLUMN]
        .apply(normalize_duplicate_value)
    )

    temp = temp[temp[DUPLICATE_GROUP_COLUMN] != ""]

    if temp.empty:
        return out

    for group_key, group_df in temp.groupby(DUPLICATE_GROUP_COLUMN):
        out[group_key] = infer_redundancy_type_for_group(group_df)

    return out


def redundancy_factor_for_type(redundancy_type: str) -> float:
    """
    Return multiplicative factor for HC redundancy type.
    """
    key = redundancy_type or "unknown"
    return float(HC_REDUNDANCY_FACTOR.get(key, HC_REDUNDANCY_FACTOR["unknown"]))


def add_hc_redundancy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add HC redundancy classification and factor.

    Important HC logic:
    - cross_year_recurring = 1.00
    - same_year duplicates get penalty
    - exact duplicates get stronger penalty
    """
    out = df.copy()

    if out.empty:
        out["hc_redundancy_type"] = []
        out["hc_redundancy_factor"] = []
        return out

    group_map = build_duplicate_group_redundancy_map(out)

    out["hc_redundancy_type"] = out.apply(
        lambda row: classify_row_redundancy(row, group_map),
        axis=1,
    )

    out["hc_redundancy_factor"] = out["hc_redundancy_type"].apply(
        redundancy_factor_for_type
    )

    return out


def summarize_redundancy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize redundancy classifications.
    """
    if "hc_redundancy_type" not in df.columns:
        df = add_hc_redundancy_columns(df)

    summary = (
        df.groupby("hc_redundancy_type", dropna=False)
        .agg(
            evidence_count=("chunk_id", "count"),
            avg_factor=("hc_redundancy_factor", "mean"),
        )
        .reset_index()
        .sort_values("evidence_count", ascending=False)
    )

    return summary


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "chunk_id": ["a", "b", "c", "d"],
            "year": [2021, 2022, 2022, 2022],
            "duplicate_group": ["g1", "g1", "g2", "g2"],
            "similarity_scope": ["cross_year_recurring", "cross_year_recurring", "near_same_year", "near_same_year"],
        }
    )

    out = add_hc_redundancy_columns(sample)
    print(out.to_string(index=False))
    print(summarize_redundancy(out).to_string(index=False))
