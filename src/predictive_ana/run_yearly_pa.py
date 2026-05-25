from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictive_ana.common import (
    attach_target_year,
    build_company_year_targets,
    drop_missing_years,
)
from purpose_articulation.pa_config import INPUT_CHUNKS_PATH, LLM_PROVIDER, OUTPUT_DIR
from purpose_articulation.pa_exporter import export_all
from purpose_articulation.pa_llm_runner import PAEvaluator
from purpose_articulation.pa_loader import load_chunks, summarize_loaded_chunks
from purpose_articulation.run_pa_score import build_diagnostics, score_company


def filter_company_year(df: pd.DataFrame, company: str, target_year: int) -> pd.DataFrame:
    return df[
        (df["company"].astype(str).str.strip() == str(company).strip())
        & (df["year"].astype("Int64") == int(target_year))
    ].copy()


def run_yearly_pa_pipeline(
    input_path: str | Path = INPUT_CHUNKS_PATH,
    output_dir: str | Path = Path(OUTPUT_DIR).parent / "predictive_pa",
    provider: str = LLM_PROVIDER,
    max_targets: int | None = None,
    company_filter: list[str] | None = None,
    year_filter: list[int] | None = None,
    min_company_year_chunks: int = 1,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print("=" * 80)
    print("Running yearly PA scoring")
    print("=" * 80)
    print(f"Input      : {input_path}")
    print(f"Output dir : {output_dir}")
    print(f"Provider   : {provider}")

    df = load_chunks(input_path)
    df = drop_missing_years(df)
    loaded_summary = summarize_loaded_chunks(df)

    targets = build_company_year_targets(
        df,
        company_filter=company_filter,
        year_filter=year_filter,
    )
    if max_targets is not None:
        targets = targets.head(max_targets).copy()

    evaluator = PAEvaluator(provider=provider)

    company_score_records: list[dict] = []
    question_score_records: list[dict] = []
    evidence_detail_records: list[dict] = []
    evidence_library_records: list[dict] = []
    raw_llm_records: list[dict] = []
    skipped_targets: list[dict] = []

    for idx, row in targets.iterrows():
        company = str(row["company"])
        target_year = int(row["year"])
        company_df = filter_company_year(df, company, target_year)

        if len(company_df) < min_company_year_chunks:
            skipped_targets.append(
                {
                    "company": company,
                    "target_year": target_year,
                    "reason": (
                        "fewer than "
                        f"min_company_year_chunks={min_company_year_chunks}"
                    ),
                    "chunk_count": len(company_df),
                }
            )
            continue

        print(
            f"[{idx + 1}/{len(targets)}] "
            f"Scoring company-year: {company} {target_year} "
            f"({len(company_df)} chunks)"
        )

        try:
            (
                evidence_detail,
                question_scores,
                company_score,
                raw_llm,
                evidence_library,
            ) = score_company(company, company_df, evaluator)

            company_score = dict(company_score)
            company_score["target_year"] = target_year

            company_score_records.append(company_score)
            question_score_records.extend(attach_target_year(question_scores, target_year))
            evidence_detail_records.extend(attach_target_year(evidence_detail, target_year))
            raw_llm_records.extend(attach_target_year(raw_llm, target_year))
            evidence_library_records.extend(attach_target_year(evidence_library, target_year))

        except Exception as exc:
            skipped_targets.append(
                {
                    "company": company,
                    "target_year": target_year,
                    "reason": f"error: {exc}",
                    "chunk_count": len(company_df),
                }
            )
            print(f"  ERROR: {company} {target_year}: {exc}")

    diagnostics_lines = build_diagnostics(
        input_path=str(input_path),
        provider=provider,
        loaded_summary=loaded_summary,
        company_score_records=company_score_records,
        question_score_records=question_score_records,
        evidence_detail_records=evidence_detail_records,
        raw_llm_records=raw_llm_records,
        evidence_library_records=evidence_library_records,
        skipped_companies=skipped_targets,
    )
    diagnostics_lines.insert(0, "YEARLY VARIANT: company-year targets; evidence restricted to target year")
    diagnostics_lines.append("")
    diagnostics_lines.append(f"company_year_targets        : {len(targets)}")

    output_paths = export_all(
        company_score_records=company_score_records,
        question_score_records=question_score_records,
        evidence_detail_records=evidence_detail_records,
        evidence_library_records=evidence_library_records,
        raw_llm_records=raw_llm_records,
        diagnostics_lines=diagnostics_lines,
        output_dir=output_dir,
    )

    print("")
    print("Yearly PA scoring complete")
    print(f"Targets scored : {len(company_score_records)}")
    print(f"Targets skipped: {len(skipped_targets)}")

    return {
        "company_scores": pd.DataFrame(company_score_records),
        "question_scores": pd.DataFrame(question_score_records),
        "evidence_details": pd.DataFrame(evidence_detail_records),
        "output_paths": output_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run company-year PA scoring.")
    parser.add_argument("--input", type=str, default=str(INPUT_CHUNKS_PATH))
    parser.add_argument("--output-dir", type=str, default="outputs/predictive_pa")
    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_PROVIDER,
        choices=["mock", "openai", "gemini", "claude"],
    )
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--company", nargs="*", default=None)
    parser.add_argument("--year", nargs="*", type=int, default=None)
    parser.add_argument("--min-company-year-chunks", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_yearly_pa_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        provider=args.provider,
        max_targets=args.max_targets,
        company_filter=args.company,
        year_filter=args.year,
        min_company_year_chunks=args.min_company_year_chunks,
    )


if __name__ == "__main__":
    main()
