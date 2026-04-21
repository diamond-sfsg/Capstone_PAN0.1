from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_score_report(df_all: pd.DataFrame, df_kept: pd.DataFrame) -> str:
    lines: list[str] = []

    lines.append("PURPOSE ARTICULATION EVIDENCE RETRIEVAL REPORT")
    lines.append("=" * 80)
    lines.append(f"total_input_rows          : {len(df_all)}")
    lines.append(f"kept_rows                 : {len(df_kept)}")
    lines.append("")

    if "retrieval_status" in df_all.columns:
        lines.append("RETRIEVAL STATUS COUNTS")
        lines.append("=" * 80)
        status_counts = df_all["retrieval_status"].value_counts(dropna=False)
        for k, v in status_counts.items():
            lines.append(f"{str(k):25s}: {int(v)}")
        lines.append("")

    score_cols = ["lexical_score", "tfidf_score", "embedding_score", "metadata_score"]
    lines.append("SCORE SUMMARY")
    lines.append("=" * 80)
    for col in score_cols:
        if col in df_kept.columns:
            s = df_kept[col]
            lines.append(f"{col}")
            lines.append(f"  min   : {s.min():.6f}")
            lines.append(f"  p25   : {s.quantile(0.25):.6f}")
            lines.append(f"  median: {s.median():.6f}")
            lines.append(f"  mean  : {s.mean():.6f}")
            lines.append(f"  p75   : {s.quantile(0.75):.6f}")
            lines.append(f"  max   : {s.max():.6f}")
            lines.append("")

    for col in score_cols:
        if col in df_kept.columns:
            lines.append(f"TOP 10 BY {col.upper()}")
            lines.append("=" * 80)
            top_df = df_kept.sort_values(col, ascending=False).head(10)
            for _, row in top_df.iterrows():
                text_preview = str(row.get("text_clean", ""))[:160].replace("\n", " ")
                lines.append(
                    f"[{row.get('chunk_id','')}] "
                    f"{row.get('company','')} | {row.get('year','')} | "
                    f"{row.get('source','')} | {row.get('section','')} | "
                    f"{col}={row.get(col, 0):.6f}"
                )
                lines.append(f"  {text_preview}")
            lines.append("")

    if "source" in df_kept.columns:
        lines.append("KEPT ROWS BY SOURCE")
        lines.append("=" * 80)
        source_counts = df_kept["source"].value_counts(dropna=False)
        for k, v in source_counts.items():
            lines.append(f"{str(k):25s}: {int(v)}")
        lines.append("")

    return "\n".join(lines)


def export_report(report_text: str, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")