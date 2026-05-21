# src/strategy_alignment/run_sa_score.py

"""
Main runner for Strategy & Source Alignment scoring.

Recommended execution:

    python src/strategy_alignment/run_sa_score.py --mock-llm

For real LLM scoring, pass an llm_client programmatically through run_sa_pipeline().
This file keeps the model client abstract so it can work with Gemini, OpenAI,
or any other provider.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategy_alignment.sa_config import (
    INPUT_CHUNKS_PATH,
    PA_PURPOSE_REFERENCE_PATH,
    PHASE_OUTPUT_DIR,
    QUESTION_ORDER,
)
from strategy_alignment.sa_loader import (
    load_all_sa_inputs,
    build_company_sa_input,
)
from strategy_alignment.sa_retrieval import (
    retrieve_company_candidates,
    retrieve_company_candidates_long,
)
from strategy_alignment.sa_prompt_builder import build_sa_batch_prompts
from strategy_alignment.sa_llm_runner import run_sa_batch_llm
from strategy_alignment.sa_aggregator import (
    aggregate_company_sa_result,
    flatten_evidence_map,
)
from strategy_alignment.sa_exporter import export_sa_outputs


LLMClient = Callable[[str], str]


# ============================================================
# Mock LLM for dry run
# ============================================================

def mock_llm_client(prompt: str) -> str:
    """
    Mock LLM client for pipeline testing.

    This does NOT produce valid research scores.
    It only verifies that the pipeline can run end-to-end.
    """
    if "Question ID: SA_Q1" in prompt:
        question_id = "SA_Q1"
        score = 2.5
        label = "Mock moderate capital alignment"
        summary = (
            "Mock result: evidence appears partially related to stated purpose, "
            "but explicit linkage is not verified."
        )
    elif "Question ID: SA_Q2" in prompt:
        question_id = "SA_Q2"
        score = 2.5
        label = "Mock moderate operational alignment"
        summary = (
            "Mock result: operational evidence appears partially related to stated "
            "purpose, but explicit linkage is not verified."
        )
    else:
        question_id = "UNKNOWN"
        score = 0.0
        label = "Unknown question"
        summary = "Mock result failed to identify question."

    return json.dumps(
        {
            "question_id": question_id,
            "llm_score_0_5": score,
            "score_label": label,
            "alignment_summary": summary,
            "purpose_connection_type": "reasonable_semantic_alignment",
            "best_supporting_evidence_ids": [],
            "contradictory_or_weak_evidence_ids": [],
            "needs_human_review": True,
            "human_review_reason": "mock_llm_for_pipeline_test_only",
        }
    )


# ============================================================
# Main pipeline
# ============================================================

def run_single_company_sa(
    company: str,
    chunks_df: pd.DataFrame,
    purpose_df: pd.DataFrame,
    llm_client: LLMClient,
    export_candidates: bool = True,
) -> Dict[str, Any]:
    """
    Run SA scoring for one company.
    """
    company_input = build_company_sa_input(
        chunks_df=chunks_df,
        purpose_df=purpose_df,
        company=company,
    )

    company_chunks = company_input["chunks"]
    purpose_reference = company_input["purpose_reference"]

    candidate_map = retrieve_company_candidates(
        company_chunks=company_chunks,
        purpose_reference=purpose_reference,
    )

    prompts = build_sa_batch_prompts(
        company=company,
        purpose_reference=purpose_reference,
        candidate_map=candidate_map,
    )

    llm_results = run_sa_batch_llm(
        prompts=prompts,
        llm_client=llm_client,
    )

    aggregated = aggregate_company_sa_result(
        company=company,
        purpose_reference=purpose_reference,
        candidate_map=candidate_map,
        llm_results=llm_results,
    )

    final_score_row = aggregated["final_score_row"]
    question_results = aggregated["question_results"]
    evidence_map = aggregated["evidence_map"]

    question_rows = []
    for question_id in QUESTION_ORDER:
        row = dict(question_results.get(question_id, {}))
        row["company"] = company
        question_rows.append(row)

    evidence_df = flatten_evidence_map(
        company=company,
        purpose_reference=purpose_reference,
        evidence_map=evidence_map,
    )

    if export_candidates:
        candidate_df = retrieve_company_candidates_long(
            company=company,
            company_chunks=company_chunks,
            purpose_reference=purpose_reference,
        )
    else:
        candidate_df = pd.DataFrame()

    return {
        "company": company,
        "final_score_row": final_score_row,
        "question_rows": question_rows,
        "evidence_df": evidence_df,
        "candidate_df": candidate_df,
        "llm_results": llm_results,
    }


def run_sa_pipeline(
    llm_client: Optional[LLMClient] = None,
    chunks_path=INPUT_CHUNKS_PATH,
    purpose_reference_path=PA_PURPOSE_REFERENCE_PATH,
    output_dir=PHASE_OUTPUT_DIR,
    max_companies: Optional[int] = None,
    company_filter: Optional[List[str]] = None,
    export_candidates: bool = True,
    print_diagnostics: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run full Strategy & Source Alignment scoring pipeline.

    Parameters:
    - llm_client:
        Callable that takes prompt string and returns model response string.
    - max_companies:
        Useful for pilot runs.
    - company_filter:
        Optional explicit list of companies to score.
    - export_candidates:
        Whether to export pre-LLM candidate retrieval details.
    """
    if llm_client is None:
        raise ValueError(
            "llm_client must be provided. "
            "For test runs, use run_sa_pipeline(llm_client=mock_llm_client)."
        )

    loaded = load_all_sa_inputs(
        chunks_path=chunks_path,
        purpose_reference_path=purpose_reference_path,
    )

    chunks_df = loaded["chunks_df"]
    purpose_df = loaded["purpose_df"]
    targets_df = loaded["targets_df"]

    if company_filter:
        company_set = set(company_filter)
        targets_df = targets_df[targets_df["company"].isin(company_set)].copy()

    targets_df = targets_df.sort_values("company").reset_index(drop=True)

    if max_companies is not None:
        targets_df = targets_df.head(max_companies).copy()

    score_rows = []
    question_rows = []
    evidence_frames = []
    candidate_frames = []

    print("=" * 80)
    print("RUNNING STRATEGY & SOURCE ALIGNMENT PIPELINE")
    print("=" * 80)
    print(f"targets: {len(targets_df)}")
    print(f"chunks: {len(chunks_df)}")
    print(f"purpose_references: {len(purpose_df)}")

    for idx, row in targets_df.iterrows():
        company = row["company"]

        print(f"[{idx + 1}/{len(targets_df)}] Scoring company: {company}")

        try:
            result = run_single_company_sa(
                company=company,
                chunks_df=chunks_df,
                purpose_df=purpose_df,
                llm_client=llm_client,
                export_candidates=export_candidates,
            )

            score_rows.append(result["final_score_row"])
            question_rows.extend(result["question_rows"])

            if not result["evidence_df"].empty:
                evidence_frames.append(result["evidence_df"])

            if export_candidates and not result["candidate_df"].empty:
                candidate_frames.append(result["candidate_df"])

        except Exception as exc:
            print(f"ERROR scoring {company}: {type(exc).__name__}: {exc}")

            score_rows.append(
                {
                    "company": company,
                    "extracted_purpose": "",
                    "purpose_statement_normalized": "",
                    "purpose_statement_raw": "",
                    "purpose_confidence_0_1": 0.0,
                    "sa_q1_score_0_5": 0.0,
                    "sa_q2_score_0_5": 0.0,
                    "sa_final_score_0_5": 0.0,
                    "sa_score_0_100": 0.0,
                    "sa_needs_human_review": True,
                    "sa_review_reason": (
                        f"pipeline_error:{type(exc).__name__}:{str(exc)}"
                    ),
                }
            )

            for question_id in QUESTION_ORDER:
                question_rows.append(
                    {
                        "company": company,
                        "question_id": question_id,
                        "question_name": "",
                        "question_score_0_5": 0.0,
                        "num_evidence": 0,
                        "best_evidence_id": "",
                        "second_best_evidence_id": "",
                        "needs_human_review": True,
                        "review_reason": (
                            f"pipeline_error:{type(exc).__name__}:{str(exc)}"
                        ),
                    }
                )

    outputs = export_sa_outputs(
        score_rows=score_rows,
        question_rows=question_rows,
        evidence_frames=evidence_frames,
        candidate_frames=candidate_frames if export_candidates else None,
        output_dir=output_dir,
        print_diagnostics=print_diagnostics,
    )

    return outputs


def _load_anthropic_key_from_local_config() -> str:
    try:
        from configs.config import ANTHROPIC_API_KEY as local_key
    except Exception:
        return ""

    return str(local_key or "").strip()


def build_claude_llm_client(
    model: str = "claude-opus-4-1-20250805",
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> LLMClient:
    api_key = (
        os.getenv("ANTHROPIC_API_KEY", "").strip()
        or _load_anthropic_key_from_local_config()
    )

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing.")

    def client(prompt: str) -> str:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": (
                "You are a careful corporate strategy alignment scoring analyst. "
                "Return valid JSON only."
            ),
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "x-api-key": api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Claude API HTTP {exc.code}: {body}") from exc

        content = data.get("content", [])
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(texts).strip()

    return client


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Strategy & Source Alignment scoring pipeline."
    )

    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM client for end-to-end pipeline testing.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "claude"],
        help="LLM provider for CLI runs.",
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("CLAUDE_MODEL", "claude-opus-4-1-20250805"),
        help="LLM model name for real provider runs.",
    )

    parser.add_argument(
        "--max-companies",
        type=int,
        default=None,
        help="Limit number of companies for pilot run.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CHUNKS_PATH,
        help="Input unified chunks CSV.",
    )

    parser.add_argument(
        "--purpose-reference",
        type=Path,
        default=PA_PURPOSE_REFERENCE_PATH,
        help="PA purpose reference file. Can be PA evidence detail CSV with extracted_purpose.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PHASE_OUTPUT_DIR,
        help="Output directory for SA files.",
    )

    parser.add_argument(
        "--company",
        nargs="*",
        default=None,
        help="Optional company name filter. Example: --company Apple Microsoft",
    )

    parser.add_argument(
        "--no-candidates",
        action="store_true",
        help="Do not export pre-LLM candidate retrieval details.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print diagnostics report to console.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mock_llm or args.provider == "mock":
        client = mock_llm_client
    elif args.provider == "claude":
        client = build_claude_llm_client(model=args.llm_model)
    else:
        raise ValueError(
            "No real LLM client is configured in run_sa_score.py. "
            "Use --mock-llm for dry run, or import run_sa_pipeline() "
            "and pass your own llm_client callable."
        )

    run_sa_pipeline(
        llm_client=client,
        chunks_path=args.input,
        purpose_reference_path=args.purpose_reference,
        output_dir=args.output_dir,
        max_companies=args.max_companies,
        company_filter=args.company,
        export_candidates=not args.no_candidates,
        print_diagnostics=not args.quiet,
    )


if __name__ == "__main__":
    main()
