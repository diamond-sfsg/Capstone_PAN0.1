from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# Allow direct execution:
# python src/purpose_articulation/run_pa_score.py
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from purpose_articulation.pa_aggregator import (  # noqa: E402
    aggregate_company_pa_score,
    aggregate_evidence_question_score,
    aggregate_q3_score,
    attach_company_score_metadata,
    compute_pa_evidence_contribution,
)
from purpose_articulation.pa_config import (  # noqa: E402
    INPUT_CHUNKS_PATH,
    LLM_PROVIDER,
    OUTPUT_DIR,
    PIPELINE_VERSION,
)
from purpose_articulation.pa_evidence_score import compute_q3_evidence_set_quality  # noqa: E402
from purpose_articulation.pa_exporter import export_all  # noqa: E402
from purpose_articulation.pa_llm_runner import PAEvaluator  # noqa: E402
from purpose_articulation.pa_loader import (  # noqa: E402
    build_company_targets,
    filter_company,
    load_chunks,
    summarize_loaded_chunks,
)
from purpose_articulation.pa_overlap import apply_overlap_to_evidence_rows  # noqa: E402
from purpose_articulation.pa_retrieval import (  # noqa: E402
    build_q3_evidence_set,
    retrieve_candidates_for_question,
    select_llm_evidence_q1_q2,
)
from purpose_articulation.pa_rubric import PA_QUESTIONS  # noqa: E402
from purpose_articulation.pa_year_stats import (  # noqa: E402
    compute_source_mix,
    compute_year_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run company-level pooled-year Purpose Articulation scoring."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_CHUNKS_PATH),
        help="Input unified chunks CSV path.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_PROVIDER,
        choices=["mock", "openai", "gemini", "claude"],
        help="LLM provider. Use mock for local pipeline testing.",
    )

    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Optional single-company filter.",
    )

    parser.add_argument(
        "--max-companies",
        type=int,
        default=None,
        help="Optional maximum number of companies for testing.",
    )

    parser.add_argument(
        "--min-company-chunks",
        type=int,
        default=1,
        help="Skip companies with fewer than this many chunks.",
    )

    return parser.parse_args()


def _safe_year(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_flags(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return "|".join(str(item) for item in value)

    if isinstance(value, str):
        return value

    return str(value)


def _row_to_evidence_record(
    *,
    company: str,
    row: pd.Series,
    question_id: str,
    scoring_type: str,
    llm_result: dict | None = None,
    raw_contribution_fields: bool = True,
) -> dict:
    llm_result = llm_result or {}

    return {
        "company": company,
        "dimension": "purpose_articulation",
        "question_id": question_id,
        "question_name": PA_QUESTIONS[question_id].name,
        "question_text": PA_QUESTIONS[question_id].question,
        "scoring_type": scoring_type,
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "source": row.get("source"),
        "normalized_source": row.get("normalized_source"),
        "section": row.get("section"),
        "subsection": row.get("subsection"),
        "year": _safe_year(row.get("year")),
        "text_clean": row.get("text_clean"),
        "candidate_rank": int(row.get("candidate_rank", 0) or 0),
        "llm_input_rank": int(row.get("llm_input_rank", 0) or 0)
        if raw_contribution_fields and row.get("llm_input_rank", None) is not None
        else None,
        "evidence_set_rank": int(row.get("evidence_set_rank", 0) or 0)
        if row.get("evidence_set_rank", None) is not None
        else None,
        "keyword_relevance": _safe_float(row.get("keyword_relevance"), 0.0),
        "context_completeness": _safe_float(row.get("context_completeness"), 0.0),
        "rag_similarity": _safe_float(row.get("rag_similarity"), 0.0),
        "base_evidence_score_0_1": _safe_float(row.get("base_evidence_score_0_1"), 0.0),
        "source_prior": _safe_float(row.get("source_prior"), 1.0),
        "source_adjusted_rank_score": _safe_float(row.get("source_adjusted_rank_score"), 0.0),
        "evidence_quality_factor": _safe_float(row.get("evidence_quality_factor"), 0.70),
        "llm_score_0_5": _safe_float(llm_result.get("llm_score_0_5"), 0.0),
        "pa_tone_bonus": _safe_float(
            llm_result.get("pa_tone_bonus", row.get("rule_based_pa_tone_bonus", 0.0)),
            0.0,
        ),
        "support_level": str(llm_result.get("support_level", "")),
        "extracted_purpose": str(llm_result.get("extracted_purpose", "")),
        "llm_reason": str(llm_result.get("reason", "")),
        "risk_flags": _serialize_flags(llm_result.get("risk_flags")),
        "overlap_count": None,
        "overlap_factor": None,
        "pa_evidence_contribution": None,
    }


def _rows_to_evidence_library_records(
    *,
    company: str,
    question_id: str,
    scoring_type: str,
    retrieval_stage: str,
    rows_df: pd.DataFrame,
) -> list[dict]:
    if rows_df is None or rows_df.empty:
        return []

    records: list[dict] = []

    for _, row in rows_df.iterrows():
        records.append(
            {
                "company": company,
                "dimension": "purpose_articulation",
                "question_id": question_id,
                "question_name": PA_QUESTIONS[question_id].name,
                "question_text": PA_QUESTIONS[question_id].question,
                "scoring_type": scoring_type,
                "retrieval_stage": retrieval_stage,
                "chunk_id": row.get("chunk_id"),
                "doc_id": row.get("doc_id"),
                "source": row.get("source"),
                "normalized_source": row.get("normalized_source"),
                "section": row.get("section"),
                "subsection": row.get("subsection"),
                "year": _safe_year(row.get("year")),
                "text_clean": row.get("text_clean"),
                "candidate_rank": int(row.get("candidate_rank", 0) or 0),
                "llm_input_rank": int(row.get("llm_input_rank", 0) or 0)
                if row.get("llm_input_rank", None) is not None
                else None,
                "evidence_set_rank": int(row.get("evidence_set_rank", 0) or 0)
                if row.get("evidence_set_rank", None) is not None
                else None,
                "keyword_relevance": _safe_float(row.get("keyword_relevance"), 0.0),
                "context_completeness": _safe_float(row.get("context_completeness"), 0.0),
                "rag_similarity": _safe_float(row.get("rag_similarity"), 0.0),
                "base_evidence_score_0_1": _safe_float(
                    row.get("base_evidence_score_0_1"),
                    0.0,
                ),
                "source_prior": _safe_float(row.get("source_prior"), 1.0),
                "source_adjusted_rank_score": _safe_float(
                    row.get("source_adjusted_rank_score"),
                    0.0,
                ),
                "evidence_quality_factor": _safe_float(
                    row.get("evidence_quality_factor"),
                    0.70,
                ),
                "rule_based_pa_tone_bonus": _safe_float(
                    row.get("rule_based_pa_tone_bonus"),
                    0.0,
                ),
            }
        )

    return records


def score_company(
    company: str,
    company_df: pd.DataFrame,
    evaluator: PAEvaluator,
) -> tuple[list[dict], list[dict], dict, list[dict], list[dict]]:
    """
    Score one company.

    Returns:
        evidence_detail_records
        question_score_records
        company_score_record
        raw_llm_records
        evidence_library_records
    """
    evidence_detail_records: list[dict] = []
    question_score_records: list[dict] = []
    raw_llm_records: list[dict] = []
    evidence_library_records: list[dict] = []

    # =========================================================================
    # Q1 / Q2: evidence-level LLM scoring
    # =========================================================================

    evidence_level_records: list[dict] = []

    for question_id in ["PA_Q1", "PA_Q2"]:
        candidates = retrieve_candidates_for_question(
            company_df=company_df,
            question_id=question_id,
        )
        evidence_library_records.extend(
            _rows_to_evidence_library_records(
                company=company,
                question_id=question_id,
                scoring_type="evidence",
                retrieval_stage="candidate_pool",
                rows_df=candidates,
            )
        )

        llm_evidence = select_llm_evidence_q1_q2(candidates)
        evidence_library_records.extend(
            _rows_to_evidence_library_records(
                company=company,
                question_id=question_id,
                scoring_type="evidence",
                retrieval_stage="llm_selected",
                rows_df=llm_evidence,
            )
        )

        for rank, (_, row) in enumerate(llm_evidence.iterrows(), start=1):
            row = row.copy()
            row["llm_input_rank"] = rank

            evidence_payload = row.to_dict()
            evidence_payload["company"] = company

            llm_result, raw_output = evaluator.score_evidence(
                question_id=question_id,
                evidence_row=evidence_payload,
            )

            evidence_record = _row_to_evidence_record(
                company=company,
                row=row,
                question_id=question_id,
                scoring_type="evidence",
                llm_result=llm_result,
            )

            evidence_level_records.append(evidence_record)

            raw_llm_records.append(
                {
                    "company": company,
                    "question_id": question_id,
                    "question_name": PA_QUESTIONS[question_id].name,
                    "scoring_type": "evidence",
                    "chunk_id": row.get("chunk_id"),
                    "provider": evaluator.provider,
                    "raw_output": raw_output,
                    "parsed_output": llm_result,
                }
            )

    evidence_level_df = pd.DataFrame(evidence_level_records)

    if not evidence_level_df.empty:
        evidence_level_df = apply_overlap_to_evidence_rows(evidence_level_df)
        evidence_level_df["pa_evidence_contribution"] = evidence_level_df.apply(
            compute_pa_evidence_contribution,
            axis=1,
        )

        evidence_detail_records.extend(
            evidence_level_df.to_dict(orient="records")
        )

    # Aggregate Q1 / Q2.
    for question_id in ["PA_Q1", "PA_Q2"]:
        q_record = aggregate_evidence_question_score(
            evidence_df=evidence_level_df,
            question_id=question_id,
        )
        q_record["company"] = company
        question_score_records.append(q_record)

    # =========================================================================
    # Q3: evidence-set LLM scoring
    # =========================================================================

    q3_set_df = build_q3_evidence_set(company_df)
    evidence_library_records.extend(
        _rows_to_evidence_library_records(
            company=company,
            question_id="PA_Q3",
            scoring_type="evidence_set",
            retrieval_stage="evidence_set_selected",
            rows_df=q3_set_df,
        )
    )
    q3_set_quality = compute_q3_evidence_set_quality(q3_set_df)

    q3_result, q3_raw_output = evaluator.score_evidence_set(
        question_id="PA_Q3",
        evidence_set_df=q3_set_df,
        set_quality=q3_set_quality,
    )

    q3_question_record = aggregate_q3_score(
        company=company,
        llm_set_result=q3_result,
        set_quality=q3_set_quality,
        evidence_set_df=q3_set_df,
    )

    question_score_records.append(q3_question_record)

    raw_llm_records.append(
        {
            "company": company,
            "question_id": "PA_Q3",
            "question_name": PA_QUESTIONS["PA_Q3"].name,
            "scoring_type": "evidence_set",
            "chunk_id": None,
            "provider": evaluator.provider,
            "raw_output": q3_raw_output,
            "parsed_output": q3_result,
            "evidence_set_quality": q3_set_quality,
        }
    )

    if q3_set_df is not None and not q3_set_df.empty:
        for _, row in q3_set_df.iterrows():
            q3_evidence_record = _row_to_evidence_record(
                company=company,
                row=row,
                question_id="PA_Q3",
                scoring_type="evidence_set",
                llm_result={
                    "support_level": q3_result.get("support_level", ""),
                    "extracted_purpose": q3_result.get("extracted_purpose", ""),
                    "reason": q3_result.get("reason", ""),
                    "risk_flags": q3_result.get("risk_flags", []),
                },
                raw_contribution_fields=False,
            )

            q3_evidence_record.update(
                {
                    "llm_set_score_0_5": _safe_float(
                        q3_result.get("llm_set_score_0_5"),
                        0.0,
                    ),
                    "source_diversity": _safe_float(
                        q3_set_quality.get("source_diversity"),
                        0.0,
                    ),
                    "formal_document_presence": _safe_float(
                        q3_set_quality.get("formal_document_presence"),
                        0.0,
                    ),
                    "strategic_section_presence": _safe_float(
                        q3_set_quality.get("strategic_section_presence"),
                        0.0,
                    ),
                    "evidence_set_quality": _safe_float(
                        q3_set_quality.get("evidence_set_quality"),
                        0.0,
                    ),
                    "evidence_set_quality_factor": _safe_float(
                        q3_set_quality.get("evidence_set_quality_factor"),
                        0.75,
                    ),
                }
            )

            evidence_detail_records.append(q3_evidence_record)

    # =========================================================================
    # Company-level aggregation
    # =========================================================================

    company_score = aggregate_company_pa_score(
        company=company,
        question_rows=question_score_records,
    )

    evidence_detail_df = pd.DataFrame(evidence_detail_records)

    year_stats = compute_year_stats(evidence_detail_df)
    source_mix = compute_source_mix(evidence_detail_df)

    company_score = attach_company_score_metadata(
        company_score=company_score,
        year_stats=year_stats,
        source_mix=source_mix,
    )

    return (
        evidence_detail_records,
        question_score_records,
        company_score,
        raw_llm_records,
        evidence_library_records,
    )


def build_diagnostics(
    *,
    input_path: str,
    provider: str,
    loaded_summary: dict,
    company_score_records: list[dict],
    question_score_records: list[dict],
    evidence_detail_records: list[dict],
    raw_llm_records: list[dict],
    evidence_library_records: list[dict],
    skipped_companies: list[dict],
) -> list[str]:
    company_df = pd.DataFrame(company_score_records)
    question_df = pd.DataFrame(question_score_records)
    evidence_df = pd.DataFrame(evidence_detail_records)

    lines = [
        "PA COMPANY-LEVEL POOLED-YEAR SCORING DIAGNOSTICS",
        "=" * 80,
        f"pipeline_version           : {PIPELINE_VERSION}",
        f"input_path                 : {input_path}",
        f"llm_provider               : {provider}",
        "",
        "Loaded chunk summary",
        "-" * 80,
        f"total_chunks               : {loaded_summary.get('total_chunks')}",
        f"total_companies            : {loaded_summary.get('total_companies')}",
        f"source_counts              : {loaded_summary.get('source_counts')}",
        f"year_min                   : {loaded_summary.get('year_min')}",
        f"year_max                   : {loaded_summary.get('year_max')}",
        f"missing_year_count         : {loaded_summary.get('missing_year_count')}",
        f"empty_section_count        : {loaded_summary.get('empty_section_count')}",
        "",
        "Scoring summary",
        "-" * 80,
        f"companies_scored           : {len(company_score_records)}",
        f"companies_skipped          : {len(skipped_companies)}",
        f"question_rows              : {len(question_score_records)}",
        f"evidence_detail_rows       : {len(evidence_detail_records)}",
        f"evidence_library_rows      : {len(evidence_library_records)}",
        f"raw_llm_rows               : {len(raw_llm_records)}",
    ]

    if not company_df.empty and "PA_score_0_100" in company_df.columns:
        lines.extend(
            [
                f"PA_score_mean_0_100       : {company_df['PA_score_0_100'].mean():.4f}",
                f"PA_score_median_0_100     : {company_df['PA_score_0_100'].median():.4f}",
                f"PA_score_min_0_100        : {company_df['PA_score_0_100'].min():.4f}",
                f"PA_score_max_0_100        : {company_df['PA_score_0_100'].max():.4f}",
            ]
        )

    if not question_df.empty and "question_id" in question_df.columns:
        lines.append("")
        lines.append("Question score means")
        lines.append("-" * 80)

        means = (
            question_df.groupby("question_id")["question_score_0_5"]
            .mean()
            .to_dict()
        )

        for question_id, mean_score in means.items():
            lines.append(f"{question_id:<28}: {mean_score:.4f}")

    if not evidence_df.empty and "normalized_source" in evidence_df.columns:
        lines.append("")
        lines.append("Scored evidence source distribution")
        lines.append("-" * 80)

        source_counts = evidence_df["normalized_source"].value_counts().to_dict()
        for source, count in source_counts.items():
            lines.append(f"{source:<28}: {count}")

    if skipped_companies:
        lines.append("")
        lines.append("Skipped companies")
        lines.append("-" * 80)
        for item in skipped_companies[:30]:
            lines.append(f"{item.get('company')}: {item.get('reason')}")

        if len(skipped_companies) > 30:
            lines.append(f"... {len(skipped_companies) - 30} more skipped companies")

    return lines


def main() -> None:
    args = parse_args()

    input_path = args.input
    output_dir = args.output_dir
    provider = args.provider

    print("=" * 80)
    print("Running PA company-level pooled-year scoring")
    print("=" * 80)
    print(f"Input      : {input_path}")
    print(f"Output dir : {output_dir}")
    print(f"Provider   : {provider}")

    df = load_chunks(input_path)
    loaded_summary = summarize_loaded_chunks(df)

    if args.company:
        targets = [args.company]
    else:
        targets = build_company_targets(df)

    if args.max_companies is not None:
        targets = targets[: args.max_companies]

    evaluator = PAEvaluator(provider=provider)

    all_company_score_records: list[dict] = []
    all_question_score_records: list[dict] = []
    all_evidence_detail_records: list[dict] = []
    all_evidence_library_records: list[dict] = []
    all_raw_llm_records: list[dict] = []
    skipped_companies: list[dict] = []

    for idx, company in enumerate(targets, start=1):
        company_df = filter_company(df, company)

        if len(company_df) < args.min_company_chunks:
            skipped_companies.append(
                {
                    "company": company,
                    "reason": f"fewer than min_company_chunks={args.min_company_chunks}",
                    "chunk_count": len(company_df),
                }
            )
            continue

        print(f"[{idx}/{len(targets)}] Scoring company: {company} ({len(company_df)} chunks)")

        try:
            (
                evidence_detail_records,
                question_score_records,
                company_score_record,
                raw_llm_records,
                evidence_library_records,
            ) = score_company(
                company=company,
                company_df=company_df,
                evaluator=evaluator,
            )

            all_evidence_detail_records.extend(evidence_detail_records)
            all_question_score_records.extend(question_score_records)
            all_company_score_records.append(company_score_record)
            all_raw_llm_records.extend(raw_llm_records)
            all_evidence_library_records.extend(evidence_library_records)

        except Exception as exc:
            skipped_companies.append(
                {
                    "company": company,
                    "reason": f"error: {exc}",
                    "chunk_count": len(company_df),
                }
            )
            print(f"  ERROR: {company}: {exc}")

    diagnostics_lines = build_diagnostics(
        input_path=input_path,
        provider=provider,
        loaded_summary=loaded_summary,
        company_score_records=all_company_score_records,
        question_score_records=all_question_score_records,
        evidence_detail_records=all_evidence_detail_records,
        raw_llm_records=all_raw_llm_records,
        evidence_library_records=all_evidence_library_records,
        skipped_companies=skipped_companies,
    )

    output_paths = export_all(
        company_score_records=all_company_score_records,
        question_score_records=all_question_score_records,
        evidence_detail_records=all_evidence_detail_records,
        evidence_library_records=all_evidence_library_records,
        raw_llm_records=all_raw_llm_records,
        diagnostics_lines=diagnostics_lines,
        output_dir=output_dir,
    )

    print("")
    print("=" * 80)
    print("PA scoring complete")
    print("=" * 80)
    for name, path in output_paths.items():
        print(f"{name:<18}: {path}")

    print("")
    print(f"Companies scored : {len(all_company_score_records)}")
    print(f"Companies skipped: {len(skipped_companies)}")


if __name__ == "__main__":
    main()
