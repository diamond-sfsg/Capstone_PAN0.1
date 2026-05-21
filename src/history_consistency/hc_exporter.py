# src/history_consistency/hc_exporter.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from history_consistency.hc_config import (
    HC_COMPANY_SCORE_OUTPUT,
    HC_EVIDENCE_OUTPUT,
HC_DIAGNOSTICS_OUTPUT,
    OUTPUT_COLUMNS,
)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _format_source_counts(df: pd.DataFrame) -> list[str]:
    if df.empty or "source" not in df.columns:
        return []

    lines = ["", "Evidence source counts", "-" * 80]
    for source, count in df["source"].fillna("unknown").value_counts().items():
        lines.append(f"{source:<28}: {int(count)}")
    return lines


def _format_score_summary(company_scores_df: pd.DataFrame) -> list[str]:
    if company_scores_df.empty or "hc_final_score_0_5" not in company_scores_df.columns:
        return []

    scores = pd.to_numeric(company_scores_df["hc_final_score_0_5"], errors="coerce").dropna()
    if scores.empty:
        return []

    return [
        "",
        "Company score summary",
        "-" * 80,
        f"mean_0_5                   : {scores.mean():.4f}",
        f"median_0_5                 : {scores.median():.4f}",
        f"min_0_5                    : {scores.min():.4f}",
        f"max_0_5                    : {scores.max():.4f}",
    ]


def build_hc_diagnostics_lines(
    company_scores_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    evidence_library_df: pd.DataFrame | None = None,
) -> list[str]:
    lines = [
        "HISTORY CONSISTENCY SCORING DIAGNOSTICS",
        "=" * 80,
        f"companies_scored           : {len(company_scores_df)}",
        f"evidence_rows              : {len(evidence_df)}",
    ]

    if evidence_library_df is not None:
        lines.append(f"evidence_library_rows      : {len(evidence_library_df)}")

    if not company_scores_df.empty:
        if "needs_human_review" in company_scores_df.columns:
            needs_review = int(company_scores_df["needs_human_review"].fillna(False).sum())
            lines.append(f"companies_needing_review   : {needs_review}")

        if "distinct_year_count" in company_scores_df.columns:
            years = pd.to_numeric(company_scores_df["distinct_year_count"], errors="coerce").dropna()
            if not years.empty:
                lines.append(f"distinct_year_mean         : {years.mean():.4f}")
                lines.append(f"distinct_year_min          : {int(years.min())}")
                lines.append(f"distinct_year_max          : {int(years.max())}")

    lines.extend(_format_score_summary(company_scores_df))
    lines.extend(_format_source_counts(evidence_df))

    if not evidence_df.empty and "hc_llm_score_source" in evidence_df.columns:
        lines.extend(["", "LLM score source counts", "-" * 80])
        for source, count in evidence_df["hc_llm_score_source"].fillna("unknown").value_counts().items():
            lines.append(f"{source:<28}: {int(count)}")

    return lines


def write_diagnostics(lines: Iterable[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(line) for line in lines), encoding="utf-8")
    return path


def export_all_hc_outputs(
    *,
    company_scores_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    output_dir: Path | str,
    evidence_library_df: pd.DataFrame | None = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    company_path = output_dir / HC_COMPANY_SCORE_OUTPUT.name
    evidence_path = output_dir / HC_EVIDENCE_OUTPUT.name
    evidence_library_path = output_dir / "hc_evidence_library_v1.csv"
    diagnostics_path = output_dir / HC_DIAGNOSTICS_OUTPUT.name

    company_out = company_scores_df.copy()
    if not company_out.empty:
        ordered = [col for col in OUTPUT_COLUMNS if col in company_out.columns]
        rest = [col for col in company_out.columns if col not in ordered]
        company_out = company_out[ordered + rest]

    output_paths = {
        "company_scores": _write_csv(company_out, company_path),
        "evidence_details": _write_csv(evidence_df, evidence_path),
    }

    if evidence_library_df is not None:
        output_paths["evidence_library"] = _write_csv(
            evidence_library_df,
            evidence_library_path,
        )

    output_paths["diagnostics"] = write_diagnostics(
        build_hc_diagnostics_lines(
            company_scores_df,
            evidence_df,
            evidence_library_df=evidence_library_df,
        ),
        diagnostics_path,
    )

    return output_paths
