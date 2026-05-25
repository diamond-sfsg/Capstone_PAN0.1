from __future__ import annotations

import argparse
import os
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

from predictive_ana.common import build_company_year_targets, drop_missing_years
from strategy_alignment.run_sa_score import (
    build_claude_llm_client,
    build_openai_llm_client,
    mock_llm_client,
)
from strategy_alignment.sa_aggregator import aggregate_company_sa_result, flatten_evidence_map
from strategy_alignment.sa_config import INPUT_CHUNKS_PATH, PA_PURPOSE_REFERENCE_PATH, QUESTION_ORDER
from strategy_alignment.sa_exporter import export_sa_outputs
from strategy_alignment.sa_loader import (
    _clip,
    _safe_float,
    _safe_string,
    build_purpose_reference_from_pa_evidence_detail,
    load_chunks,
)
from strategy_alignment.sa_llm_runner import run_sa_batch_llm
from strategy_alignment.sa_prompt_builder import build_sa_batch_prompts
from strategy_alignment.sa_retrieval import retrieve_company_candidates, retrieve_company_candidates_long


def load_yearly_purpose_reference(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "target_year" not in df.columns and "year" in df.columns:
        df["target_year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "target_year" not in df.columns:
        raise ValueError("Yearly SA requires PA evidence detail with target_year.")

    df = df[df["target_year"].notna()].copy()
    refs = []

    for target_year, group in df.groupby("target_year"):
        ref = build_purpose_reference_from_pa_evidence_detail(group)
        if ref.empty:
            continue
        ref["target_year"] = int(target_year)
        refs.append(ref)

    if not refs:
        return pd.DataFrame()

    out = pd.concat(refs, ignore_index=True)
    out["company"] = out["company"].map(_safe_string)
    out["purpose_statement_normalized"] = (
        out["purpose_statement_normalized"].fillna("").astype(str).str.strip()
    )
    out["purpose_confidence_0_1"] = out["purpose_confidence_0_1"].map(
        lambda x: _clip(_safe_float(x, 0.0), 0.0, 1.0)
    )
    out = out[
        out["company"].ne("")
        & out["purpose_statement_normalized"].ne("")
        & out["target_year"].notna()
    ].copy()
    out["target_year"] = out["target_year"].astype(int)
    return out.reset_index(drop=True)


def get_yearly_purpose_reference(
    purpose_df: pd.DataFrame,
    company: str,
    target_year: int,
) -> dict[str, Any]:
    match = purpose_df[
        (purpose_df["company"].astype(str) == str(company))
        & (purpose_df["target_year"].astype(int) == int(target_year))
    ]
    if match.empty:
        return {
            "company": company,
            "purpose_statement_normalized": "",
            "purpose_confidence_0_1": 0.0,
        }
    row = match.iloc[0].to_dict()
    row["purpose_confidence_0_1"] = _clip(
        _safe_float(row.get("purpose_confidence_0_1", 0.0)),
        0.0,
        1.0,
    )
    return row


def make_llm_client(provider: str, model: str | None = None):
    if provider == "mock":
        return mock_llm_client
    if provider == "openai":
        return build_openai_llm_client(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "claude":
        return build_claude_llm_client(
            model=model or os.getenv("CLAUDE_MODEL", "claude-opus-4-1-20250805")
        )
    raise ValueError(f"Unsupported provider: {provider}")


def run_yearly_sa_pipeline(
    input_path: str | Path = INPUT_CHUNKS_PATH,
    purpose_reference_path: str | Path = PA_PURPOSE_REFERENCE_PATH,
    output_dir: str | Path = "outputs/predictive_sa",
    provider: str = "mock",
    llm_model: str | None = None,
    max_targets: int | None = None,
    company_filter: list[str] | None = None,
    year_filter: list[int] | None = None,
    export_candidates: bool = True,
    print_diagnostics: bool = True,
) -> dict[str, pd.DataFrame]:
    chunks_df = drop_missing_years(load_chunks(Path(input_path)))
    purpose_df = load_yearly_purpose_reference(purpose_reference_path)

    targets = build_company_year_targets(
        chunks_df,
        company_filter=company_filter,
        year_filter=year_filter,
    )
    targets = targets.merge(
        purpose_df[["company", "target_year", "purpose_statement_normalized"]],
        left_on=["company", "year"],
        right_on=["company", "target_year"],
        how="inner",
    ).drop(columns=["target_year"])
    targets = targets.sort_values(["company", "year"]).reset_index(drop=True)
    if max_targets is not None:
        targets = targets.head(max_targets).copy()

    llm_client = make_llm_client(provider, llm_model)

    score_rows: list[dict] = []
    question_rows: list[dict] = []
    evidence_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []

    print("=" * 80)
    print("RUNNING YEARLY STRATEGY & SOURCE ALIGNMENT PIPELINE")
    print("=" * 80)
    print(f"targets: {len(targets)}")
    print(f"chunks: {len(chunks_df)}")
    print(f"purpose_references: {len(purpose_df)}")

    for idx, row in targets.iterrows():
        company = str(row["company"])
        target_year = int(row["year"])
        company_chunks = chunks_df[
            (chunks_df["company"].astype(str) == company)
            & (chunks_df["year"].astype(int) == target_year)
        ].copy()
        purpose_reference = get_yearly_purpose_reference(purpose_df, company, target_year)

        print(f"[{idx + 1}/{len(targets)}] Scoring company-year: {company} {target_year}")

        try:
            candidate_map = retrieve_company_candidates(
                company_chunks=company_chunks,
                purpose_reference=purpose_reference,
            )
            prompts = build_sa_batch_prompts(
                company=company,
                purpose_reference=purpose_reference,
                candidate_map=candidate_map,
            )
            llm_results = run_sa_batch_llm(prompts=prompts, llm_client=llm_client)
            aggregated = aggregate_company_sa_result(
                company=company,
                purpose_reference=purpose_reference,
                candidate_map=candidate_map,
                llm_results=llm_results,
            )

            final_score = dict(aggregated["final_score_row"])
            final_score["target_year"] = target_year
            score_rows.append(final_score)

            for question_id in QUESTION_ORDER:
                question_row = dict(aggregated["question_results"].get(question_id, {}))
                question_row["company"] = company
                question_row["target_year"] = target_year
                question_rows.append(question_row)

            evidence_df = flatten_evidence_map(
                company=company,
                purpose_reference=purpose_reference,
                evidence_map=aggregated["evidence_map"],
            )
            if not evidence_df.empty:
                evidence_df["target_year"] = target_year
                evidence_frames.append(evidence_df)

            if export_candidates:
                candidate_df = retrieve_company_candidates_long(
                    company=company,
                    company_chunks=company_chunks,
                    purpose_reference=purpose_reference,
                )
                if not candidate_df.empty:
                    candidate_df["target_year"] = target_year
                    candidate_frames.append(candidate_df)

        except Exception as exc:
            print(f"ERROR scoring {company} {target_year}: {type(exc).__name__}: {exc}")
            score_rows.append(
                {
                    "company": company,
                    "target_year": target_year,
                    "sa_final_score_0_5": 0.0,
                    "sa_score_0_100": 0.0,
                    "sa_needs_human_review": True,
                    "sa_review_reason": f"pipeline_error:{type(exc).__name__}:{exc}",
                }
            )

    return export_sa_outputs(
        score_rows=score_rows,
        question_rows=question_rows,
        evidence_frames=evidence_frames,
        candidate_frames=candidate_frames if export_candidates else None,
        output_dir=Path(output_dir),
        print_diagnostics=print_diagnostics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run company-year SA scoring.")
    parser.add_argument("--input", type=str, default=str(INPUT_CHUNKS_PATH))
    parser.add_argument("--purpose-reference", type=str, default=str(PA_PURPOSE_REFERENCE_PATH))
    parser.add_argument("--output-dir", type=str, default="outputs/predictive_sa")
    parser.add_argument("--provider", choices=["mock", "openai", "claude"], default="mock")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--company", nargs="*", default=None)
    parser.add_argument("--year", nargs="*", type=int, default=None)
    parser.add_argument("--no-candidates", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_yearly_sa_pipeline(
        input_path=args.input,
        purpose_reference_path=args.purpose_reference,
        output_dir=args.output_dir,
        provider=args.provider,
        llm_model=args.llm_model,
        max_targets=args.max_targets,
        company_filter=args.company,
        year_filter=args.year,
        export_candidates=not args.no_candidates,
        print_diagnostics=not args.quiet,
    )


if __name__ == "__main__":
    main()
