"""
Phase 2A: Evidence Scoring Pipeline

This module implements the evidence scoring phase of the purpose-driven company scoring pipeline.
It processes unified text chunks from company filings and computes relevance scores across three
purpose dimensions (Purpose Articulation, History Consistency, Strategy Alignment) using multiple
scoring methods: lexical matching, TF-IDF similarity, embedding similarity, metadata matching,
and prompt-pattern matching.

Input: data/clean_2.0/unified_chunks_v4.csv
Output: data/phase2/evidence_score_v1_newdata.csv

The output scores (e.g., pa_sum_score, hc_sum_score, sa_sum_score) are used in Phase 2B for
bucket assignment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_score_v1.collect_scores import compute_dimension_score_frame, merge_dimension_frames
from evidence_score_v1.config import (
    DEFAULT_DIAGNOSTICS_TXT,
    DEFAULT_INPUT_CSV,
    EVIDENCE_MATRIX_OUTPUT_PATH,
    PHASE2_DIR,
)
from evidence_score_v1.data_loader import load_chunk_corpus
from evidence_score_v1.dimension_registry import get_dimension_config, get_dimension_prefix
from evidence_score_v1.text_normalize import add_retrieval_text_columns


OUTPUT_CSV = EVIDENCE_MATRIX_OUTPUT_PATH
OUTPUT_DIAGNOSTICS = DEFAULT_DIAGNOSTICS_TXT


def _write_diagnostics(df: pd.DataFrame, output_path: Path) -> None:
    score_cols = [
        "pa_lexical_score",
        "pa_tfidf_score",
        "pa_embedding_score",
        "pa_metadata_score",
        "pa_prompt_score",
        "pa_sum_score",
        "hc_lexical_score",
        "hc_tfidf_score",
        "hc_embedding_score",
        "hc_metadata_score",
        "hc_prompt_score",
        "hc_history_bonus_score",
        "hc_sum_score",
        "sa_lexical_score",
        "sa_tfidf_score",
        "sa_embedding_score",
        "sa_metadata_score",
        "sa_prompt_score",
        "sa_sum_score",
    ]

    lines: list[str] = []
    lines.append("EVIDENCE SCORE V1 DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"rows: {len(df)}")
    lines.append(f"columns: {len(df.columns)}")
    lines.append("")

    available_score_cols = [c for c in score_cols if c in df.columns]
    if available_score_cols:
        lines.append("SCORE COLUMN SUMMARY")
        lines.append("=" * 80)
        summary = df[available_score_cols].describe().T
        lines.append(summary.to_string())
        lines.append("")

        lines.append("TOP 10 BY pa_sum_score")
        lines.append("=" * 80)
        top_pa = df.sort_values("pa_sum_score", ascending=False).head(10)
        lines.append(
            top_pa[
                ["chunk_id", "company", "year", "source", "section", "pa_sum_score"]
            ].to_string(index=False)
        )
        lines.append("")

        lines.append("TOP 10 BY hc_sum_score")
        lines.append("=" * 80)
        top_hc = df.sort_values("hc_sum_score", ascending=False).head(10)
        lines.append(
            top_hc[
                ["chunk_id", "company", "year", "source", "section", "hc_sum_score"]
            ].to_string(index=False)
        )
        lines.append("")

        lines.append("TOP 10 BY sa_sum_score")
        lines.append("=" * 80)
        top_sa = df.sort_values("sa_sum_score", ascending=False).head(10)
        lines.append(
            top_sa[
                ["chunk_id", "company", "year", "source", "section", "sa_sum_score"]
            ].to_string(index=False)
        )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)

    base_df = load_chunk_corpus(DEFAULT_INPUT_CSV)
    base_df = add_retrieval_text_columns(base_df)

    pa_cfg = get_dimension_config("purpose_articulation")
    hc_cfg = get_dimension_config("history_consistency")
    sa_cfg = get_dimension_config("strategy_alignment")

    pa_prefix = get_dimension_prefix("purpose_articulation")
    hc_prefix = get_dimension_prefix("history_consistency")
    sa_prefix = get_dimension_prefix("strategy_alignment")

    pa_frame = compute_dimension_score_frame(
        base_df,
        dim_cfg=pa_cfg,
        prefix=pa_prefix,
        include_history_bonus=False,
    )
    hc_frame = compute_dimension_score_frame(
        base_df,
        dim_cfg=hc_cfg,
        prefix=hc_prefix,
        include_history_bonus=True,
    )
    sa_frame = compute_dimension_score_frame(
        base_df,
        dim_cfg=sa_cfg,
        prefix=sa_prefix,
        include_history_bonus=False,
    )

    final_df = merge_dimension_frames(
        base_df=base_df,
        score_frames=[pa_frame, hc_frame, sa_frame],
    )

    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    _write_diagnostics(final_df, OUTPUT_DIAGNOSTICS)

    print(f"[DONE] wrote CSV: {OUTPUT_CSV}")
    print(f"[DONE] wrote diagnostics: {OUTPUT_DIAGNOSTICS}")


if __name__ == "__main__":
    main()
