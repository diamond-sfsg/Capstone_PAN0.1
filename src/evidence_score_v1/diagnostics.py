from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def build_basic_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    diagnostics: Dict[str, object] = {
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    for col in [
        "source",
        "section",
        "company",
        "year",
    ]:
        if col in df.columns:
            diagnostics[f"{col}_non_null"] = int(df[col].notna().sum())
            diagnostics[f"{col}_unique"] = int(df[col].nunique(dropna=True))

    for score_col in [
        "lexical_raw_score",
        "tfidf_cosine",
        "embedding_cosine",
        "section_prior_score",
        "source_prior_score",
        "metadata_total_score",
        "prompt_match_score",
        "diagnostic_rank_score",
    ]:
        if score_col in df.columns:
            diagnostics[f"{score_col}_mean"] = float(df[score_col].fillna(0).mean())
            diagnostics[f"{score_col}_max"] = float(df[score_col].fillna(0).max())

    return diagnostics


def format_diagnostics_report(
    diagnostics: Dict[str, object],
    title: str = "Evidence Score Diagnostics",
) -> str:
    lines = [title.upper(), "=" * 40]
    for key, value in diagnostics.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def write_diagnostics(
    df: pd.DataFrame,
    path: str | Path,
    title: str = "Evidence Score Diagnostics",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    diagnostics = build_basic_diagnostics(df)
    report_text = format_diagnostics_report(diagnostics, title=title)

    path.write_text(report_text, encoding="utf-8")
