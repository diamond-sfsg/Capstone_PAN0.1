from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from purpose_articulation.pa_config import (
    OUTPUT_DIR,
    PA_COMPANY_SCORE_PATH,
    PA_DIAGNOSTICS_PATH,
    PA_EVIDENCE_DETAIL_PATH,
    PA_EVIDENCE_LIBRARY_PATH,
    PA_LLM_RAW_OUTPUT_PATH,
    PA_QUESTION_SCORE_PATH,
)


def ensure_output_dir(output_dir: str | Path = OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _json_default(obj: Any) -> str:
    return str(obj)


def export_csv(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


def export_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            )


def export_diagnostics(lines: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def resolve_output_paths(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Path]:
    """
    Resolve output paths.

    If output_dir is the default OUTPUT_DIR, this still uses the standard config paths.
    If user passes a custom output_dir, filenames remain the same under that directory.
    """
    output_dir = Path(output_dir)

    if output_dir.resolve() == Path(OUTPUT_DIR).resolve():
        return {
            "company_score": PA_COMPANY_SCORE_PATH,
            "question_score": PA_QUESTION_SCORE_PATH,
            "evidence_detail": PA_EVIDENCE_DETAIL_PATH,
            "evidence_library": PA_EVIDENCE_LIBRARY_PATH,
            "llm_raw": PA_LLM_RAW_OUTPUT_PATH,
            "diagnostics": PA_DIAGNOSTICS_PATH,
        }

    return {
        "company_score": output_dir / "pa_company_score_v1.csv",
        "question_score": output_dir / "pa_question_score_v1.csv",
        "evidence_detail": output_dir / "pa_evidence_detail_v1.csv",
        "evidence_library": output_dir / "pa_evidence_library_v1.csv",
        "llm_raw": output_dir / "pa_llm_raw_outputs_v1.jsonl",
        "diagnostics": output_dir / "pa_diagnostics_v1.txt",
    }


def export_all(
    company_score_records: list[dict],
    question_score_records: list[dict],
    evidence_detail_records: list[dict],
    evidence_library_records: list[dict],
    raw_llm_records: list[dict],
    diagnostics_lines: list[str],
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir)
    paths = resolve_output_paths(output_dir)

    export_csv(evidence_library_records, paths["evidence_library"])
    export_csv(company_score_records, paths["company_score"])
    export_csv(question_score_records, paths["question_score"])
    export_csv(evidence_detail_records, paths["evidence_detail"])
    export_jsonl(raw_llm_records, paths["llm_raw"])
    export_diagnostics(diagnostics_lines, paths["diagnostics"])

    return paths
