"""
Phase 2B: Evidence Bucket Assignment

This module assigns each scored evidence chunk to the most relevant purpose dimension bucket
based on scoring thresholds. It categorizes chunks into Purpose Articulation, History Consistency,
or Strategy Alignment buckets, handling overlaps and edge cases with review flags.

Input: data/phase2/evidence_score_v1_newdata.csv (from Phase 2A)
Output: data/phase2/evidence_buckets_v1/*_evidence_v1_newdata.csv (per-dimension buckets)

The bucketed evidence is then passed to Phase 3 for LLM quality review.
"""

from __future__ import annotations

import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from evidence_buckets_v1.evidence_bucket_config import (
    INPUT_FILE,
    OUTPUT_DIR,
    BUCKET_CONFIG,
    AMBIGUITY_MARGIN,
    ID_COL,
    DUPLICATE_GROUP_COL,
    BASE_KEEP_COLS,
    SCORE_KEEP_COLS,
)


def validate_input(df: pd.DataFrame) -> None:
    required = [ID_COL]
    required += [cfg["score_col"] for cfg in BUCKET_CONFIG.values()]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[ID_COL].isna().any():
        raise ValueError(f"{ID_COL} contains missing values.")

    duplicated = df[ID_COL].duplicated().sum()
    if duplicated:
        print(f"[WARN] {duplicated:,} duplicated chunk_id values found.")


def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def build_membership_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for bucket_name, cfg in BUCKET_CONFIG.items():
        short = cfg["short"]
        score_col = cfg["score_col"]
        threshold = cfg["threshold"]

        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

        df[f"{short}_selected"] = df[score_col] >= threshold
        df[f"{short}_threshold"] = threshold
        df[f"{short}_score_percentile"] = df[score_col].rank(pct=True)

    selected_cols = [
        f"{cfg['short']}_selected"
        for cfg in BUCKET_CONFIG.values()
    ]

    df["bucket_overlap_count"] = df[selected_cols].sum(axis=1)

    def make_overlap_type(row):
        active = []
        for _, cfg in BUCKET_CONFIG.items():
            short = cfg["short"]
            if row[f"{short}_selected"]:
                active.append(short.upper())

        if not active:
            return "NONE"

        return "_".join(active)

    df["bucket_overlap_type"] = df.apply(make_overlap_type, axis=1)

    score_cols = [cfg["score_col"] for cfg in BUCKET_CONFIG.values()]
    scores = df[score_cols]

    df["top_score"] = scores.max(axis=1)
    df["second_score"] = scores.apply(
        lambda row: row.nlargest(2).iloc[-1],
        axis=1,
    )

    score_col_to_bucket = {
        cfg["score_col"]: bucket_name
        for bucket_name, cfg in BUCKET_CONFIG.items()
    }

    df["top_bucket_by_score"] = scores.idxmax(axis=1).map(score_col_to_bucket)
    df["top_score_margin"] = df["top_score"] - df["second_score"]

    df["needs_overlap_review"] = df["bucket_overlap_count"] > 1
    df["needs_margin_review"] = (
        (df["bucket_overlap_count"] >= 1)
        & (df["top_score_margin"] < AMBIGUITY_MARGIN)
    )

    df["review_flag"] = "ok"
    df.loc[df["needs_overlap_review"], "review_flag"] = "overlap"
    df.loc[df["needs_margin_review"], "review_flag"] = "low_margin"
    df.loc[
        df["needs_overlap_review"] & df["needs_margin_review"],
        "review_flag",
    ] = "overlap_and_low_margin"

    return df


def export_bucket_files(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    keep_cols = safe_cols(df, BASE_KEEP_COLS + SCORE_KEEP_COLS)

    diagnostic_cols = safe_cols(
        df,
        [
            "bucket_overlap_count",
            "bucket_overlap_type",
            "top_bucket_by_score",
            "top_score",
            "second_score",
            "top_score_margin",
            "needs_overlap_review",
            "needs_margin_review",
            "review_flag",
        ],
    )

    for bucket_name, cfg in BUCKET_CONFIG.items():
        short = cfg["short"]
        score_col = cfg["score_col"]

        bucket_df = df[df[f"{short}_selected"]].copy()

        bucket_df["evidence_bucket"] = bucket_name
        bucket_df["bucket_score"] = bucket_df[score_col]
        bucket_df["bucket_threshold"] = cfg["threshold"]

        export_cols = (
            ["evidence_bucket", "bucket_score", "bucket_threshold"]
            + keep_cols
            + diagnostic_cols
        )

        bucket_df = bucket_df[export_cols].sort_values(
            by=["bucket_score", "top_score_margin"],
            ascending=[False, True],
        )

        csv_path = OUTPUT_DIR / cfg["output_csv"]
        db_path = OUTPUT_DIR / cfg["output_db"]

        bucket_df.to_csv(csv_path, index=False)

        with sqlite3.connect(db_path) as conn:
            bucket_df.to_sql(
                "evidence",
                conn,
                if_exists="replace",
                index=False,
            )

        print(f"[EXPORT] {bucket_name}: {len(bucket_df):,} rows")
        print(f"         CSV: {csv_path}")
        print(f"         DB : {db_path}")


def pairwise_overlap_report(df: pd.DataFrame) -> list[str]:
    lines = []

    lines.append("PAIRWISE EXACT CHUNK OVERLAP")
    lines.append("=" * 80)

    bucket_items = list(BUCKET_CONFIG.items())

    for (name_a, cfg_a), (name_b, cfg_b) in combinations(bucket_items, 2):
        flag_a = f"{cfg_a['short']}_selected"
        flag_b = f"{cfg_b['short']}_selected"

        set_a = set(df.loc[df[flag_a], ID_COL])
        set_b = set(df.loc[df[flag_b], ID_COL])

        intersection = set_a & set_b
        union = set_a | set_b

        jaccard = len(intersection) / len(union) if union else 0
        containment_a = len(intersection) / len(set_a) if set_a else 0
        containment_b = len(intersection) / len(set_b) if set_b else 0

        lines.append(f"{name_a} vs {name_b}")
        lines.append(f"  {name_a} count        : {len(set_a):,}")
        lines.append(f"  {name_b} count        : {len(set_b):,}")
        lines.append(f"  intersection          : {len(intersection):,}")
        lines.append(f"  union                 : {len(union):,}")
        lines.append(f"  jaccard               : {jaccard:.4f}")
        lines.append(f"  containment in {name_a}: {containment_a:.4f}")
        lines.append(f"  containment in {name_b}: {containment_b:.4f}")
        lines.append("")

    return lines


def duplicate_group_overlap_report(df: pd.DataFrame) -> list[str]:
    lines = []

    lines.append("PAIRWISE DUPLICATE-GROUP OVERLAP")
    lines.append("=" * 80)

    if DUPLICATE_GROUP_COL not in df.columns:
        lines.append("Skipped: duplicate_group column not found.")
        lines.append("")
        return lines

    valid = df[df[DUPLICATE_GROUP_COL].notna()].copy()
    valid = valid[valid[DUPLICATE_GROUP_COL].astype(str).str.strip() != ""]

    if valid.empty:
        lines.append("No valid duplicate_group values found.")
        lines.append("")
        return lines

    bucket_items = list(BUCKET_CONFIG.items())

    for (name_a, cfg_a), (name_b, cfg_b) in combinations(bucket_items, 2):
        flag_a = f"{cfg_a['short']}_selected"
        flag_b = f"{cfg_b['short']}_selected"

        groups_a = set(valid.loc[valid[flag_a], DUPLICATE_GROUP_COL])
        groups_b = set(valid.loc[valid[flag_b], DUPLICATE_GROUP_COL])

        intersection = groups_a & groups_b
        union = groups_a | groups_b

        jaccard = len(intersection) / len(union) if union else 0

        lines.append(f"{name_a} vs {name_b}")
        lines.append(f"  duplicate groups in {name_a}: {len(groups_a):,}")
        lines.append(f"  duplicate groups in {name_b}: {len(groups_b):,}")
        lines.append(f"  group intersection         : {len(intersection):,}")
        lines.append(f"  group union                : {len(union):,}")
        lines.append(f"  group jaccard              : {jaccard:.4f}")
        lines.append("")

    return lines


def export_membership_and_review(df: pd.DataFrame) -> None:
    membership_path = OUTPUT_DIR / "evidence_bucket_membership_v1_newdata.csv"
    review_path = OUTPUT_DIR / "evidence_overlap_review_v1_newdata.csv"

    membership_cols = safe_cols(
        df,
        BASE_KEEP_COLS
        + SCORE_KEEP_COLS
        + [
            "pa_selected",
            "hc_selected",
            "sa_selected",
            "pa_threshold",
            "hc_threshold",
            "sa_threshold",
            "pa_score_percentile",
            "hc_score_percentile",
            "sa_score_percentile",
            "bucket_overlap_count",
            "bucket_overlap_type",
            "top_bucket_by_score",
            "top_score",
            "second_score",
            "top_score_margin",
            "needs_overlap_review",
            "needs_margin_review",
            "review_flag",
        ],
    )

    df[membership_cols].to_csv(membership_path, index=False)

    review_df = df[df["review_flag"] != "ok"].copy()
    review_df = review_df.sort_values(
        by=["bucket_overlap_count", "top_score_margin", "top_score"],
        ascending=[False, True, False],
    )

    review_df[membership_cols].to_csv(review_path, index=False)

    print(f"[EXPORT] membership: {membership_path}")
    print(f"[EXPORT] review    : {review_path}")


def write_diagnostics(df: pd.DataFrame) -> None:
    lines = []

    lines.append("EVIDENCE BUCKET V1 DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"input_file : {INPUT_FILE}")
    lines.append(f"output_dir : {OUTPUT_DIR}")
    lines.append(f"rows       : {len(df):,}")
    lines.append("")

    lines.append("THRESHOLDS")
    lines.append("=" * 80)

    for bucket_name, cfg in BUCKET_CONFIG.items():
        short = cfg["short"]
        score_col = cfg["score_col"]

        selected_count = int(df[f"{short}_selected"].sum())
        selected_rate = selected_count / len(df) if len(df) else 0

        lines.append(f"{bucket_name}")
        lines.append(f"  score_col : {score_col}")
        lines.append(f"  threshold : {cfg['threshold']}")
        lines.append(f"  selected  : {selected_count:,}")
        lines.append(f"  rate      : {selected_rate:.4f}")
        lines.append("")

    lines.append("OVERLAP COUNT DISTRIBUTION")
    lines.append("=" * 80)

    overlap_dist = df["bucket_overlap_count"].value_counts().sort_index()
    for overlap_count, count in overlap_dist.items():
        rate = count / len(df) if len(df) else 0
        lines.append(f"{overlap_count} bucket(s): {count:,} ({rate:.4f})")

    lines.append("")

    lines.append("OVERLAP TYPE DISTRIBUTION")
    lines.append("=" * 80)

    type_dist = df["bucket_overlap_type"].value_counts()
    for overlap_type, count in type_dist.items():
        rate = count / len(df) if len(df) else 0
        lines.append(f"{overlap_type}: {count:,} ({rate:.4f})")

    lines.append("")

    lines.extend(pairwise_overlap_report(df))
    lines.extend(duplicate_group_overlap_report(df))

    review_count = int((df["review_flag"] != "ok").sum())
    review_rate = review_count / len(df) if len(df) else 0

    lines.append("REVIEW SUMMARY")
    lines.append("=" * 80)
    lines.append(f"needs review: {review_count:,}")
    lines.append(f"review rate : {review_rate:.4f}")
    lines.append("")

    diagnostics_path = OUTPUT_DIR / "evidence_overlap_diagnostics_v1_newdata.txt"
    diagnostics_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[EXPORT] diagnostics: {diagnostics_path}")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Expected output from Phase 2 scoring: "
            "data/phase2/evidence_score_v1_newdata.csv"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    validate_input(df)

    df = build_membership_flags(df)

    export_bucket_files(df)
    export_membership_and_review(df)
    write_diagnostics(df)

    print("\nDONE: evidence bucket assignment v1 complete.")


if __name__ == "__main__":
    main()
