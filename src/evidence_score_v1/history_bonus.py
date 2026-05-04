from __future__ import annotations

import pandas as pd


def compute_history_bonus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute history-consistency bonus using existing Phase 1 fields:
    - duplicate_group
    - similarity_scope

    Assumptions:
    - cross-year recurring evidence is marked by similarity_scope == 'cross_year_recurring'
    - duplicate_group clusters similar chunks within company scope
    """

    required = {"chunk_id", "company", "year", "duplicate_group", "similarity_scope"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()

    work["is_cross_year_recurring"] = (
        work["similarity_scope"].fillna("").astype(str).str.lower() == "cross_year_recurring"
    )

    # count how many distinct years appear in each duplicate_group
    year_span_by_group = (
        work.loc[work["duplicate_group"].notna()]
        .groupby("duplicate_group")["year"]
        .nunique()
        .rename("cross_year_group_year_count")
    )

    group_size_by_group = (
        work.loc[work["duplicate_group"].notna()]
        .groupby("duplicate_group")["chunk_id"]
        .count()
        .rename("duplicate_group_size")
    )

    work = work.merge(
        year_span_by_group,
        on="duplicate_group",
        how="left",
    )
    work = work.merge(
        group_size_by_group,
        on="duplicate_group",
        how="left",
    )

    work["cross_year_group_year_count"] = work["cross_year_group_year_count"].fillna(1)
    work["duplicate_group_size"] = work["duplicate_group_size"].fillna(1)

    def bonus_fn(row) -> float:
        if not row["is_cross_year_recurring"]:
            return 0.0

        extra_years = max(int(row["cross_year_group_year_count"]) - 1, 0)

        # each extra year gives a small incremental bonus
        bonus = min(extra_years * 0.08, 0.40)
        return round(bonus, 4)

    work["history_bonus_score"] = work.apply(bonus_fn, axis=1)

    return work