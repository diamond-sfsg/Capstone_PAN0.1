# src/history_consistency/run_hc_score.py

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

from history_consistency.hc_config import (
    INPUT_CHUNKS_CSV,
    OUTPUT_DIR,
    TOP_K_EVIDENCE,
    TOP_K_EVIDENCE_PER_YEAR,
    validate_hc_config,
)
from history_consistency.hc_loader import (
    load_chunks,
    build_company_evidence_pools,
    summarize_company_pools,
)
from history_consistency.hc_retrieval import (
    retrieve_all_company_hc_candidates,
    flatten_retrieval_results,
)
from history_consistency.hc_redundancy import add_hc_redundancy_columns
from history_consistency.hc_evidence_score import score_hc_evidence_dataframe
from history_consistency.hc_aggregator import aggregate_all_company_hc_scores
from history_consistency.hc_exporter import export_all_hc_outputs
from history_consistency.hc_bonus import compute_hc_history_bonus

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def maybe_load_dotenv() -> None:
    """
    Load .env if python-dotenv is available.
    """
    if load_dotenv is not None:
        load_dotenv()


def _is_cross_year_recurring_row(row: pd.Series) -> bool:
    redundancy_type = str(row.get("hc_redundancy_type", "") or "").lower()
    similarity_scope = str(row.get("similarity_scope", "") or "").lower()

    return (
        redundancy_type == "cross_year_recurring"
        or "cross_year" in similarity_scope
        or "recurring" in similarity_scope
    )


def build_hc_evidence_library(
    evidence_df: pd.DataFrame,
    use_llm: bool,
) -> pd.DataFrame:
    """
    Build an auditable HC EvidenceLibrary from selected evidence rows.

    HC has one company-level LLM prompt when use_llm=True. When LLM is skipped,
    rows are still marked as selected retrieval evidence and as participating in
    fallback scoring / aggregation.
    """
    if evidence_df is None or evidence_df.empty:
        return pd.DataFrame()

    out = evidence_df.copy()

    out["dimension"] = "history_consistency"
    out["question_id"] = "HC_Q1"
    out["retrieval_stage"] = "selected_evidence_set"
    out["selected_for_llm"] = bool(use_llm)
    out["selected_for_llm_reason"] = (
        "Selected evidence included in the company-level HC LLM prompt."
        if use_llm
        else "LLM skipped for this run; selected evidence used by fallback scoring."
    )
    out["used_in_final_scoring"] = True
    out["used_in_base_score_aggregation"] = False
    out["base_score_component"] = ""
    out["participates_in_bonus"] = False
    out["bonus_component"] = ""

    for company, idx in out.groupby("company").groups.items():
        group = out.loc[idx].copy()

        contribution = pd.to_numeric(
            group.get("hc_evidence_contribution_0_5"),
            errors="coerce",
        )
        years = pd.to_numeric(group.get("year"), errors="coerce")

        base_components = {row_idx: [] for row_idx in group.index}

        if not contribution.dropna().empty:
            best_idx = contribution.idxmax()
            base_components[best_idx].append("best_evidence")

            best_year = years.loc[best_idx] if best_idx in years.index else None
            if pd.notna(best_year):
                other_years = group[
                    years.notna() & (years.astype(float) != float(best_year))
                ]
                other_scores = pd.to_numeric(
                    other_years.get("hc_evidence_contribution_0_5"),
                    errors="coerce",
                )
                if not other_scores.dropna().empty:
                    base_components[other_scores.idxmax()].append(
                        "best_distinct_year_evidence"
                    )

            for _, year_group in group.assign(_year_numeric=years).dropna(
                subset=["_year_numeric"]
            ).groupby("_year_numeric"):
                year_scores = pd.to_numeric(
                    year_group.get("hc_evidence_contribution_0_5"),
                    errors="coerce",
                )
                if not year_scores.dropna().empty:
                    base_components[year_scores.idxmax()].append(
                        "top_evidence_by_year"
                    )

        bonus_result = compute_hc_history_bonus(group)
        valid_year_mask = years.notna()
        distinct_year_count = int(years.dropna().astype(int).nunique())
        latest_year = int(years.dropna().max()) if not years.dropna().empty else None
        cutoff = latest_year - 3 if latest_year is not None else None

        bonus_components = {row_idx: [] for row_idx in group.index}

        for row_idx, row in group.iterrows():
            year = years.loc[row_idx] if row_idx in years.index else None

            if valid_year_mask.loc[row_idx] and bonus_result.has_multi_year_coverage:
                bonus_components[row_idx].append("multi_year_coverage")

            if valid_year_mask.loc[row_idx] and bonus_result.has_three_plus_year_coverage:
                bonus_components[row_idx].append("three_plus_year_coverage")

            if (
                bonus_result.has_cross_year_recurring_theme
                and _is_cross_year_recurring_row(row)
            ):
                bonus_components[row_idx].append("cross_year_recurring_theme")

            if (
                bonus_result.has_historical_and_recent_coverage
                and pd.notna(year)
                and cutoff is not None
            ):
                if int(year) > cutoff:
                    bonus_components[row_idx].append("recent_coverage")
                else:
                    bonus_components[row_idx].append("historical_coverage")

        for row_idx in group.index:
            base_value = "|".join(sorted(set(base_components[row_idx])))
            bonus_value = "|".join(sorted(set(bonus_components[row_idx])))

            out.at[row_idx, "base_score_component"] = base_value
            out.at[row_idx, "used_in_base_score_aggregation"] = bool(base_value)
            out.at[row_idx, "bonus_component"] = bonus_value
            out.at[row_idx, "participates_in_bonus"] = bool(bonus_value)
            out.at[row_idx, "company_bonus_distinct_year_count"] = distinct_year_count
            out.at[row_idx, "company_hc_history_bonus"] = bonus_result.bonus

    preferred = [
        "company",
        "dimension",
        "question_id",
        "retrieval_stage",
        "selected_for_llm",
        "selected_for_llm_reason",
        "used_in_final_scoring",
        "used_in_base_score_aggregation",
        "base_score_component",
        "participates_in_bonus",
        "bonus_component",
        "company_hc_history_bonus",
        "company_bonus_distinct_year_count",
        "hc_selected_rank",
        "chunk_id",
        "year",
        "source",
        "section",
        "hc_rank_score",
        "hc_base_evidence_score_0_1",
        "hc_evidence_contribution_0_5",
        "hc_redundancy_type",
        "hc_redundancy_factor",
        "text_clean",
    ]
    ordered = [col for col in preferred if col in out.columns]
    rest = [col for col in out.columns if col not in ordered]

    return out[ordered + rest]


def run_hc_pipeline(
    input_path: Path | str = INPUT_CHUNKS_CSV,
    output_dir: Path | str = OUTPUT_DIR,
    top_k: int = TOP_K_EVIDENCE,
    top_k_per_year: int = TOP_K_EVIDENCE_PER_YEAR,
    use_llm: bool = False,
    llm_provider: str = "claude",
    llm_model: str = "claude-opus-4-1-20250805",
    max_companies: Optional[int] = None,
    company_filter: Optional[list[str]] = None,
    allow_fallback_llm_score: bool = True,
) -> dict:
    """
    Run full History Consistency pipeline.

    Steps:
    1. Validate config
    2. Load unified chunks
    3. Build company-level 10-year evidence pools
    4. Retrieve selected HC candidates
    5. Optional LLM company-level rubric scoring
    6. Add redundancy factor
    7. Compute evidence contribution
    8. Aggregate company-level HC score
    9. Export outputs
    """
    validate_hc_config()

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print("=" * 80)
    print("HISTORY CONSISTENCY PIPELINE")
    print("=" * 80)
    print(f"input_path: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"use_llm: {use_llm}")
    print("")

    print("[1/8] Loading chunks...")
    chunks_df = load_chunks(input_path)
    print(f"Loaded chunks: {len(chunks_df)}")

    print("[2/8] Building company evidence pools...")
    pools = build_company_evidence_pools(chunks_df)
    pool_summary = summarize_company_pools(pools)
    print(f"Company pools: {len(pools)}")

    if company_filter:
        company_set = {str(company).strip() for company in company_filter}
        pools = {k: v for k, v in pools.items() if str(k).strip() in company_set}
        pool_summary = pool_summary[pool_summary["company"].astype(str).isin(company_set)]
        print(f"Filtered to companies: {len(pools)}")

    if max_companies is not None:
        selected_companies = pool_summary["company"].head(max_companies).tolist()
        pools = {k: v for k, v in pools.items() if k in selected_companies}
        print(f"Limited to max_companies: {len(pools)}")

    print("[3/8] Retrieving HC candidate evidence...")
    retrieval_results = retrieve_all_company_hc_candidates(
        pools=pools,
        top_k=top_k,
        top_k_per_year=top_k_per_year,
    )
    evidence_df = flatten_retrieval_results(retrieval_results)
    print(f"Selected evidence rows: {len(evidence_df)}")

    if evidence_df.empty:
        print("No evidence selected. Exporting empty outputs.")
        company_scores_df = pd.DataFrame()
        evidence_library_df = build_hc_evidence_library(evidence_df, use_llm=use_llm)
        paths = export_all_hc_outputs(
            company_scores_df=company_scores_df,
            evidence_df=evidence_df,
            evidence_library_df=evidence_library_df,
            output_dir=output_dir,
        )
        return {
            "company_scores": company_scores_df,
            "evidence_details": evidence_df,
            "output_paths": paths,
        }

    print("[4/8] Running optional LLM company-level HC rubric scoring...")
    if use_llm:
        maybe_load_dotenv()

        from history_consistency.hc_llm_runner import (
            run_hc_llm_for_all_companies,
            attach_company_llm_scores_to_evidence,
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

        print(f"LLM company results: {len(llm_results_df)}")
    else:
        print("Skipping LLM. Will use fallback score from retrieval base score.")

    print("[5/8] Adding HC redundancy factors...")
    evidence_df = add_hc_redundancy_columns(evidence_df)

    print("[6/8] Computing HC evidence contribution...")
    evidence_df = score_hc_evidence_dataframe(
        evidence_df,
        llm_score_col="hc_llm_score_0_5",
        allow_fallback_llm_score=allow_fallback_llm_score,
    )

    print("[7/8] Aggregating company-level HC scores...")
    company_scores_df = aggregate_all_company_hc_scores(evidence_df)
    print(f"Company scores: {len(company_scores_df)}")

    evidence_library_df = build_hc_evidence_library(evidence_df, use_llm=use_llm)

    print("[8/8] Exporting outputs...")
    paths = export_all_hc_outputs(
        company_scores_df=company_scores_df,
        evidence_df=evidence_df,
        evidence_library_df=evidence_library_df,
        output_dir=output_dir,
    )

    print("")
    print("Exported:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("=" * 80)

    return {
        "company_scores": company_scores_df,
        "evidence_details": evidence_df,
        "evidence_library": evidence_library_df,
        "output_paths": paths,
    }


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run History Consistency scoring pipeline."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_CHUNKS_CSV),
        help="Input unified chunks CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_EVIDENCE,
        help="Top K evidence rows per company.",
    )

    parser.add_argument(
        "--top-k-per-year",
        type=int,
        default=TOP_K_EVIDENCE_PER_YEAR,
        help="Max selected evidence rows per year.",
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Run LLM rubric scoring. If omitted, fallback score is used.",
    )

    parser.add_argument(
        "--llm-provider",
        type=str,
        default="claude",
        choices=["gemini", "claude"],
        help="LLM provider.",
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default="claude-opus-4-1-20250805",
        help="LLM model name.",
    )

    parser.add_argument(
        "--max-companies",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    parser.add_argument(
        "--company",
        nargs="*",
        default=None,
        help="Optional company name filter.",
    )

    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback score when LLM score is missing.",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()

    run_hc_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        top_k=args.top_k,
        top_k_per_year=args.top_k_per_year,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        max_companies=args.max_companies,
        company_filter=args.company,
        allow_fallback_llm_score=not args.no_fallback,
    )


if __name__ == "__main__":
    main()
