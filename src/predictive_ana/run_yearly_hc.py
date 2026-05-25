from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictive_ana.common import build_company_year_targets, drop_missing_years
from history_consistency.hc_aggregator import aggregate_all_company_hc_scores
from history_consistency.hc_config import (
    INPUT_CHUNKS_CSV,
    OUTPUT_DIR,
    TOP_K_EVIDENCE,
    TOP_K_EVIDENCE_PER_YEAR,
    validate_hc_config,
)
from history_consistency.hc_evidence_score import score_hc_evidence_dataframe
from history_consistency.hc_exporter import export_all_hc_outputs
from history_consistency.hc_loader import CompanyEvidencePool, load_chunks
from history_consistency.hc_redundancy import add_hc_redundancy_columns
from history_consistency.hc_retrieval import (
    flatten_retrieval_results,
    retrieve_all_company_hc_candidates,
)
from history_consistency.run_hc_score import build_hc_evidence_library, maybe_load_dotenv


def build_company_year_evidence_pools(
    chunks_df: pd.DataFrame,
    targets_df: pd.DataFrame,
) -> dict[str, CompanyEvidencePool]:
    pools: dict[str, CompanyEvidencePool] = {}

    for _, row in targets_df.iterrows():
        company = str(row["company"])
        target_year = int(row["year"])
        key = f"{company}__{target_year}"
        group = chunks_df[
            (chunks_df["company"].astype(str) == company)
            & (chunks_df["year"].astype(int) <= target_year)
        ].copy()
        group["source_company"] = company
        group["target_year"] = target_year
        # Existing HC aggregators group by company. Use a temporary company-year
        # key, then restore the original company after aggregation.
        group["company"] = key

        valid_years = group["year"].dropna().astype(int)
        start_year = int(valid_years.min()) if not valid_years.empty else None
        end_year = target_year

        pools[key] = CompanyEvidencePool(
            company=key,
            latest_year=target_year,
            start_year=start_year,
            end_year=end_year,
            evidence_count=len(group),
            distinct_year_count=int(valid_years.nunique()),
            data=group.reset_index(drop=True),
        )

    return pools


def restore_company_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "source_company" in out.columns:
        out["company_year_key"] = out["company"]
        out["company"] = out["source_company"]
    elif "company" in out.columns and out["company"].astype(str).str.contains("__").any():
        split = out["company"].astype(str).str.rsplit("__", n=1, expand=True)
        out["company_year_key"] = out["company"]
        out["company"] = split[0]
        out["target_year"] = pd.to_numeric(split[1], errors="coerce").astype("Int64")
    return out


def run_yearly_hc_pipeline(
    input_path: str | Path = INPUT_CHUNKS_CSV,
    output_dir: str | Path = Path(OUTPUT_DIR).parent / "predictive_hc",
    top_k: int = TOP_K_EVIDENCE,
    top_k_per_year: int = TOP_K_EVIDENCE_PER_YEAR,
    use_llm: bool = False,
    llm_provider: str = "claude",
    llm_model: str = "claude-opus-4-1-20250805",
    max_targets: Optional[int] = None,
    company_filter: Optional[list[str]] = None,
    year_filter: Optional[list[int]] = None,
    allow_fallback_llm_score: bool = True,
) -> dict:
    validate_hc_config()

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print("=" * 80)
    print("YEARLY HISTORY CONSISTENCY PIPELINE")
    print("=" * 80)
    print(f"input_path: {input_path}")
    print(f"output_dir: {output_dir}")
    print("Evidence pool rule: year <= target_year; rows without year excluded")

    chunks_df = drop_missing_years(load_chunks(input_path))
    targets = build_company_year_targets(
        chunks_df,
        company_filter=company_filter,
        year_filter=year_filter,
    )
    if max_targets is not None:
        targets = targets.head(max_targets).copy()

    print(f"company-year targets: {len(targets)}")
    pools = build_company_year_evidence_pools(chunks_df, targets)

    retrieval_results = retrieve_all_company_hc_candidates(
        pools=pools,
        top_k=top_k,
        top_k_per_year=top_k_per_year,
    )
    evidence_df = flatten_retrieval_results(retrieval_results)
    print(f"Selected evidence rows: {len(evidence_df)}")

    if not evidence_df.empty and use_llm:
        maybe_load_dotenv()
        from history_consistency.hc_llm_runner import (
            attach_company_llm_scores_to_evidence,
            run_hc_llm_for_all_companies,
        )

        llm_results_df = run_hc_llm_for_all_companies(
            evidence_df=evidence_df,
            provider=llm_provider,
            model_name=llm_model,
            max_companies=None,
        )
        evidence_df = attach_company_llm_scores_to_evidence(
            evidence_df=evidence_df,
            llm_results_df=llm_results_df,
        )

    if evidence_df.empty:
        company_scores_df = pd.DataFrame()
        evidence_library_df = build_hc_evidence_library(evidence_df, use_llm=use_llm)
    else:
        evidence_df = add_hc_redundancy_columns(evidence_df)
        evidence_df = score_hc_evidence_dataframe(
            evidence_df,
            llm_score_col="hc_llm_score_0_5",
            allow_fallback_llm_score=allow_fallback_llm_score,
        )
        company_scores_df = aggregate_all_company_hc_scores(evidence_df)
        company_scores_df = restore_company_columns(company_scores_df)
        evidence_df = restore_company_columns(evidence_df)
        evidence_library_df = build_hc_evidence_library(evidence_df, use_llm=use_llm)
        evidence_library_df = restore_company_columns(evidence_library_df)

    paths = export_all_hc_outputs(
        company_scores_df=company_scores_df,
        evidence_df=evidence_df,
        evidence_library_df=evidence_library_df,
        output_dir=output_dir,
    )

    print("Yearly HC scoring complete")
    print(f"Targets scored: {len(company_scores_df)}")
    return {
        "company_scores": company_scores_df,
        "evidence_details": evidence_df,
        "evidence_library": evidence_library_df,
        "output_paths": paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run company-year cumulative HC scoring.")
    parser.add_argument("--input", type=str, default=str(INPUT_CHUNKS_CSV))
    parser.add_argument("--output-dir", type=str, default="outputs/predictive_hc")
    parser.add_argument("--top-k", type=int, default=TOP_K_EVIDENCE)
    parser.add_argument("--top-k-per-year", type=int, default=TOP_K_EVIDENCE_PER_YEAR)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-provider", choices=["openai", "gemini", "claude"], default="claude")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--company", nargs="*", default=None)
    parser.add_argument("--year", nargs="*", type=int, default=None)
    parser.add_argument("--no-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.llm_model or (
        "gpt-4o-mini"
        if args.llm_provider == "openai"
        else "gemini-2.5-flash"
        if args.llm_provider == "gemini"
        else "claude-opus-4-1-20250805"
    )
    run_yearly_hc_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        top_k=args.top_k,
        top_k_per_year=args.top_k_per_year,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_model=model,
        max_targets=args.max_targets,
        company_filter=args.company,
        year_filter=args.year,
        allow_fallback_llm_score=not args.no_fallback,
    )


if __name__ == "__main__":
    main()
