from __future__ import annotations

import json

import pandas as pd

from purpose_articulation.pa_config import RECENT_YEAR_WINDOW


def _empty_year_stats(note: str) -> dict:
    return {
        "evidence_year_count": 0,
        "evidence_years_covered": "",
        "earliest_evidence_year": None,
        "latest_evidence_year": None,
        "evidence_year_span": 0,
        "dominant_year": None,
        "dominant_year_evidence_ratio": 0.0,
        "recent_evidence_ratio": 0.0,
        "evidence_count_by_year": "{}",
        "year_coverage_note": note,
    }


def compute_year_stats(evidence_df: pd.DataFrame) -> dict:
    """
    Compute year coverage diagnostics for pooled-year PA scoring.

    Important:
        These fields do not affect score directly.
        They are limitation / interpretability metadata.
    """
    if evidence_df is None or evidence_df.empty:
        return _empty_year_stats("No scored evidence available.")

    if "year" not in evidence_df.columns:
        return _empty_year_stats("Scored evidence does not contain year metadata.")

    years = pd.to_numeric(evidence_df["year"], errors="coerce").dropna()

    if years.empty:
        return _empty_year_stats("Scored evidence does not contain valid year metadata.")

    years = years.astype(int)

    counts = years.value_counts().sort_index()

    earliest = int(years.min())
    latest = int(years.max())
    span = int(latest - earliest + 1)

    dominant_year = int(counts.idxmax())
    dominant_year_ratio = float(counts.max() / counts.sum())

    recent_cutoff = latest - RECENT_YEAR_WINDOW + 1
    recent_ratio = float((years >= recent_cutoff).sum() / len(years))

    unique_years = sorted(years.unique().tolist())
    evidence_year_count = int(len(unique_years))
    years_covered = ",".join(str(int(year)) for year in unique_years)

    evidence_count_by_year = {
        str(int(year)): int(count)
        for year, count in counts.to_dict().items()
    }

    if evidence_year_count == 1:
        note = (
            "Evidence is concentrated in a single year; interpret this as an "
            "available-disclosure PA score rather than a longitudinal measure."
        )
    elif recent_ratio < 0.25:
        note = (
            "Limited recent evidence; score may not fully reflect current purpose "
            "articulation."
        )
    elif dominant_year_ratio >= 0.70:
        note = (
            "Evidence is concentrated in one dominant year; year coverage is uneven."
        )
    else:
        note = "Evidence is distributed across multiple years."

    return {
        "evidence_year_count": evidence_year_count,
        "evidence_years_covered": years_covered,
        "earliest_evidence_year": earliest,
        "latest_evidence_year": latest,
        "evidence_year_span": span,
        "dominant_year": dominant_year,
        "dominant_year_evidence_ratio": dominant_year_ratio,
        "recent_evidence_ratio": recent_ratio,
        "evidence_count_by_year": json.dumps(evidence_count_by_year, ensure_ascii=False),
        "year_coverage_note": note,
    }


def compute_source_mix(evidence_df: pd.DataFrame) -> str:
    """
    Compute source distribution for scored evidence.

    Returns JSON string:
        {
          "official_web": {"count": 3, "ratio": 0.5},
          "edgar": {"count": 2, "ratio": 0.33}
        }
    """
    if evidence_df is None or evidence_df.empty:
        return "{}"

    source_col = None

    if "normalized_source" in evidence_df.columns:
        source_col = "normalized_source"
    elif "source" in evidence_df.columns:
        source_col = "source"

    if source_col is None:
        return "{}"

    counts = (
        evidence_df[source_col]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
        .value_counts()
    )

    total = int(counts.sum())

    if total == 0:
        return "{}"

    source_mix = {
        str(source): {
            "count": int(count),
            "ratio": float(count / total),
        }
        for source, count in counts.to_dict().items()
    }

    return json.dumps(source_mix, ensure_ascii=False)


def compute_scored_evidence_coverage(evidence_df: pd.DataFrame) -> dict:
    """
    Combined evidence coverage diagnostics.
    """
    year_stats = compute_year_stats(evidence_df)
    source_mix = compute_source_mix(evidence_df)

    result = dict(year_stats)
    result["source_mix"] = source_mix

    return result
