from __future__ import annotations

import pandas as pd

from purpose_articulation.pa_config import OVERLAP_FACTOR_4_PLUS, OVERLAP_FACTORS


def _overlap_factor(count: int) -> float:
    if count in OVERLAP_FACTORS:
        return float(OVERLAP_FACTORS[count])
    return float(OVERLAP_FACTOR_4_PLUS)


def apply_overlap_to_evidence_rows(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply decreasing weight when the same chunk supports multiple PA questions.

    The first use of a chunk keeps full weight; repeated uses receive smaller
    factors. This prevents one strong statement from inflating Q1 and Q2.
    """
    if evidence_df is None or evidence_df.empty:
        return evidence_df.copy() if evidence_df is not None else pd.DataFrame()

    df = evidence_df.copy()

    if "chunk_id" not in df.columns:
        df["overlap_count"] = 1
        df["overlap_factor"] = 1.0
        return df

    sort_cols = [
        col
        for col in ["question_id", "llm_input_rank", "candidate_rank"]
        if col in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols).copy()

    df["overlap_count"] = df.groupby("chunk_id").cumcount() + 1
    df["overlap_factor"] = df["overlap_count"].apply(_overlap_factor)

    return df
