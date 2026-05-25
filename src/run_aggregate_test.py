from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_PA_DIR = Path("outputs/pa_test1")
DEFAULT_SA_DIR = Path("outputs/sa_test1")
DEFAULT_HC_DIR = Path("outputs/hc_test1")
DEFAULT_OUTPUT_DIR = Path("outout/aggregate/test")
DEFAULT_TEST_INPUT = Path("data/test/test1_rand/unified_chunks_final_v4.csv")
DEFAULT_PHASE_OUTPUT_ROOT = Path("outputs/aggregate_test1_rand_claude")
DEFAULT_CLAUDE_MODEL = "claude-opus-4-1-20250805"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

PA_FILES = [
    "pa_company_score_v1.csv",
    "pa_question_score_v1.csv",
    "pa_evidence_detail_v1.csv",
    "pa_evidence_library_v1.csv",
]
SA_FILES = [
    "company_sa_score_v1.csv",
    "company_sa_question_scores_v1.csv",
    "sa_evidence_details_v1.csv",
    "sa_candidate_details_v1.csv",
    "sa_evidence_library_v1.csv",
]
HC_FILES = [
    "hc_company_score_v1.csv",
    "hc_evidence_details_v1.csv",
    "hc_evidence_library_v1.csv",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {"items": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def write_progress_table(progress_path: Path, output_dir: Path) -> Path | None:
    if not progress_path.exists():
        return None

    progress = load_progress(progress_path)
    rows = list(progress.get("items", {}).values())
    if not rows:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "aggregate_run_progress.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def progress_key(company: str, phase: str) -> str:
    return f"{company}_{phase}"


def mark_progress(
    *,
    progress_path: Path,
    progress: dict,
    company: str,
    phase: str,
    status: str,
    output_dir: Path,
    message: str = "",
) -> None:
    progress.setdefault("items", {})[progress_key(company, phase)] = {
        "company": company,
        "phase": phase,
        "status": status,
        "output_dir": str(output_dir),
        "message": message,
        "updated_at": now_iso(),
    }
    save_progress(progress_path, progress)


def is_completed(progress: dict, company: str, phase: str) -> bool:
    item = progress.get("items", {}).get(progress_key(company, phase), {})
    return item.get("status") == "completed"


def run_command(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)


def selected_companies(input_path: Path) -> list[str]:
    df = pd.read_csv(input_path, usecols=["company"], dtype={"company": "string"})
    companies = (
        df["company"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted(companies[companies.ne("")].unique().tolist())


def combine_csv_outputs(run_dirs: list[Path], filenames: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        frames = []
        for run_dir in run_dirs:
            path = run_dir / filename
            if path.exists():
                frame = pd.read_csv(path, low_memory=False)
                if not frame.empty:
                    frames.append(frame)

        out_path = output_dir / filename
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                out_path,
                index=False,
                encoding="utf-8-sig",
            )
        else:
            pd.DataFrame().to_csv(out_path, index=False, encoding="utf-8-sig")


def combine_text_outputs(run_dirs: list[Path], filename: str, output_path: Path) -> None:
    parts = []
    for run_dir in run_dirs:
        path = run_dir / filename
        if path.exists():
            parts.append(f"# {run_dir}\n{path.read_text(encoding='utf-8', errors='replace')}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(parts), encoding="utf-8")


def combine_jsonl_outputs(run_dirs: list[Path], filename: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for run_dir in run_dirs:
            path = run_dir / filename
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    out.write(text)
                    out.write("\n")


def combine_phase_outputs(phase_root: Path, companies: list[str]) -> tuple[Path, Path, Path]:
    company_root = phase_root / "company_runs"
    pa_dirs = [company_root / company / "pa" for company in companies]
    sa_dirs = [company_root / company / "sa" for company in companies]
    hc_dirs = [company_root / company / "hc" for company in companies]

    pa_dir = phase_root / "pa"
    sa_dir = phase_root / "sa"
    hc_dir = phase_root / "hc"

    combine_csv_outputs(pa_dirs, PA_FILES, pa_dir)
    combine_jsonl_outputs(pa_dirs, "pa_llm_raw_outputs_v1.jsonl", pa_dir / "pa_llm_raw_outputs_v1.jsonl")
    combine_text_outputs(pa_dirs, "pa_diagnostics_v1.txt", pa_dir / "pa_diagnostics_v1.txt")

    combine_csv_outputs(sa_dirs, SA_FILES, sa_dir)
    combine_text_outputs(sa_dirs, "sa_diagnostics_v1.txt", sa_dir / "sa_diagnostics_v1.txt")

    combine_csv_outputs(hc_dirs, HC_FILES, hc_dir)
    combine_text_outputs(hc_dirs, "hc_diagnostics_v1.txt", hc_dir / "hc_diagnostics_v1.txt")

    return pa_dir, sa_dir, hc_dir


def run_scoring_phases(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_path = Path(args.input)
    phase_root = Path(args.phase_output_root)
    progress_path = Path(args.progress_path or phase_root / "progress_company_phase.json")
    companies = selected_companies(input_path)
    progress = load_progress(progress_path)
    progress.update(
        {
            "input_path": str(input_path),
            "provider": args.provider,
            "llm_model": args.llm_model,
            "phase_output_root": str(phase_root),
            "updated_at": now_iso(),
        }
    )
    save_progress(progress_path, progress)

    env = os.environ.copy()
    env["PA_LLM_FALLBACK_TO_MOCK"] = "false"
    if args.provider == "openai":
        env["OPENAI_MODEL"] = args.llm_model
    elif args.provider == "claude":
        env["CLAUDE_MODEL"] = args.llm_model

    company_root = phase_root / "company_runs"

    for index, company in enumerate(companies, start=1):
        print(f"[{index}/{len(companies)}] {company}")
        company_dir = company_root / company
        pa_dir = company_dir / "pa"
        sa_dir = company_dir / "sa"
        hc_dir = company_dir / "hc"

        if args.resume and is_completed(progress, company, "PA"):
            print(f"  skip {progress_key(company, 'PA')}")
        else:
            try:
                run_command(
                    [
                        sys.executable,
                        "src/purpose_articulation/run_pa_score.py",
                        "--input",
                        str(input_path),
                        "--output-dir",
                        str(pa_dir),
                        "--provider",
                        args.provider,
                        "--company",
                        company,
                    ],
                    env=env,
                )
                mark_progress(
                    progress_path=progress_path,
                    progress=progress,
                    company=company,
                    phase="PA",
                    status="completed",
                    output_dir=pa_dir,
                )
            except Exception as exc:
                mark_progress(
                    progress_path=progress_path,
                    progress=progress,
                    company=company,
                    phase="PA",
                    status="failed",
                    output_dir=pa_dir,
                    message=str(exc),
                )
                raise

        if args.resume and is_completed(progress, company, "SA"):
            print(f"  skip {progress_key(company, 'SA')}")
        else:
            try:
                run_command(
                    [
                        sys.executable,
                        "src/strategy_alignment/run_sa_score.py",
                        "--input",
                        str(input_path),
                        "--purpose-reference",
                        str(pa_dir / "pa_evidence_detail_v1.csv"),
                        "--output-dir",
                        str(sa_dir),
                        "--provider",
                        args.provider,
                        "--llm-model",
                        args.llm_model,
                        "--company",
                        company,
                    ],
                    env=env,
                )
                mark_progress(
                    progress_path=progress_path,
                    progress=progress,
                    company=company,
                    phase="SA",
                    status="completed",
                    output_dir=sa_dir,
                )
            except Exception as exc:
                mark_progress(
                    progress_path=progress_path,
                    progress=progress,
                    company=company,
                    phase="SA",
                    status="failed",
                    output_dir=sa_dir,
                    message=str(exc),
                )
                raise

        if args.resume and is_completed(progress, company, "HC"):
            print(f"  skip {progress_key(company, 'HC')}")
        else:
            try:
                run_command(
                    [
                        sys.executable,
                        "src/history_consistency/run_hc_score.py",
                        "--input",
                        str(input_path),
                        "--output-dir",
                        str(hc_dir),
                        "--use-llm",
                        "--llm-provider",
                        args.provider,
                        "--llm-model",
                        args.llm_model,
                        "--company",
                        company,
                    ],
                    env=env,
                )
                mark_progress(
                    progress_path=progress_path,
                    progress=progress,
                    company=company,
                    phase="HC",
                    status="completed",
                    output_dir=hc_dir,
                )
            except Exception as exc:
                mark_progress(
                    progress_path=progress_path,
                    progress=progress,
                    company=company,
                    phase="HC",
                    status="failed",
                    output_dir=hc_dir,
                    message=str(exc),
                )
                raise

    return combine_phase_outputs(phase_root, companies)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def weighted_top_two(df: pd.DataFrame, value_col: str) -> float:
    if df.empty or value_col not in df.columns:
        return 0.0

    values = (
        pd.to_numeric(df[value_col], errors="coerce")
        .dropna()
        .sort_values(ascending=False)
        .tolist()
    )

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(0.70 * values[0] + 0.30 * values[1])


def mean_numeric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def get_company_purpose(pa_detail: pd.DataFrame, sa_company: pd.DataFrame) -> Dict[str, str]:
    purpose_by_company: Dict[str, str] = {}

    if "company" in pa_detail.columns and "extracted_purpose" in pa_detail.columns:
        for company, group in pa_detail.groupby("company", dropna=False):
            values = (
                group["extracted_purpose"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            values = values[values.ne("")]
            if not values.empty:
                purpose_by_company[str(company)] = values.iloc[0]

    if "company" in sa_company.columns and "extracted_purpose" in sa_company.columns:
        for _, row in sa_company.iterrows():
            company = str(row.get("company", ""))
            if company not in purpose_by_company:
                purpose_by_company[company] = safe_str(row.get("extracted_purpose"))

    return purpose_by_company


def build_pa_components(pa_company: pd.DataFrame, pa_question: pd.DataFrame, pa_detail: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, question_row in pa_question.iterrows():
        company = safe_str(question_row.get("company"))
        question_id = safe_str(question_row.get("question_id"))
        group = pa_detail[
            (pa_detail["company"].astype(str) == company)
            & (pa_detail["question_id"].astype(str) == question_id)
        ].copy()

        if safe_str(question_row.get("scoring_type")) == "evidence_set":
            llm_rubric = safe_float(question_row.get("llm_set_score_0_5"))
            evidence_factor = safe_float(question_row.get("evidence_set_quality_factor"), 1.0)
            bonus = 0.0
            aggregation_component = {
                "evidence_set_quality": safe_float(question_row.get("evidence_set_quality")),
                "source_diversity": safe_float(question_row.get("source_diversity")),
                "formal_document_presence": safe_float(question_row.get("formal_document_presence")),
                "strategic_section_presence": safe_float(question_row.get("strategic_section_presence")),
            }
        else:
            llm_rubric = weighted_top_two(group, "llm_score_0_5")
            evidence_factor = mean_numeric(group, "evidence_quality_factor")
            bonus = weighted_top_two(group, "pa_tone_bonus")
            aggregation_component = {
                "mean_evidence_quality_factor": evidence_factor,
                "mean_overlap_factor": mean_numeric(group, "overlap_factor"),
                "top2_weighting": "0.70 * best evidence + 0.30 * second best evidence",
            }

        rows.append(
            {
                "company": company,
                "dimension": "purpose_articulation",
                "question_id": question_id,
                "question_name": question_row.get("question_name", ""),
                "llm_rubric_score_0_5": llm_rubric,
                "llm_rubric_score_source": "llm_or_mock_output",
                "evidence_factor_name": "evidence_quality_factor/evidence_set_quality_factor",
                "evidence_factor_value": evidence_factor,
                "evidence_quality_factor": mean_numeric(group, "evidence_quality_factor"),
                "overlap_factor": mean_numeric(group, "overlap_factor"),
                "llm_selection_factor": "",
                "evidence_set_quality_factor": safe_float(question_row.get("evidence_set_quality_factor"), 0.0),
                "source_diversity_factor": safe_float(question_row.get("source_diversity"), 0.0),
                "formal_document_factor": safe_float(question_row.get("formal_document_presence"), 0.0),
                "strategic_section_factor": safe_float(question_row.get("strategic_section_presence"), 0.0),
                "best_evidence_score_0_5": "",
                "best_distinct_year_evidence_score_0_5": "",
                "mean_top_evidence_by_year_0_5": "",
                "base_score_before_bonus_0_5": "",
                "bonus_name": "pa_tone_bonus" if bonus else "",
                "bonus_value_0_5": bonus,
                "final_question_score_0_5": safe_float(question_row.get("question_score_0_5")),
                "final_question_score_0_100": safe_float(question_row.get("question_score_0_100")),
                "aggregation_component_json": json.dumps(aggregation_component, ensure_ascii=False),
                "needs_human_review": bool(question_row.get("needs_human_review", False)),
            }
        )

    return pd.DataFrame(rows)


def build_sa_components(sa_company: pd.DataFrame, sa_question: pd.DataFrame, sa_detail: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, question_row in sa_question.iterrows():
        company = safe_str(question_row.get("company"))
        question_id = safe_str(question_row.get("question_id"))
        group = sa_detail[
            (sa_detail["company"].astype(str) == company)
            & (sa_detail["question_id"].astype(str) == question_id)
        ].copy()

        llm_values = pd.to_numeric(group.get("llm_score_0_5"), errors="coerce").dropna()
        llm_rubric = float(llm_values.iloc[0]) if not llm_values.empty else 0.0

        aggregation_component = {
            "best_evidence_contribution_0_5": safe_float(question_row.get("best_evidence_contribution_0_5")),
            "second_best_evidence_contribution_0_5": safe_float(question_row.get("second_best_evidence_contribution_0_5")),
            "mean_evidence_quality_factor": mean_numeric(group, "evidence_quality_factor"),
            "mean_overlap_factor": mean_numeric(group, "overlap_factor"),
            "mean_llm_selection_factor": mean_numeric(group, "llm_selection_factor"),
            "top2_weighting": "0.70 * best evidence + 0.30 * second best evidence",
        }

        rows.append(
            {
                "company": company,
                "dimension": "strategy_alignment",
                "question_id": question_id,
                "question_name": question_row.get("question_name", ""),
                "llm_rubric_score_0_5": llm_rubric,
                "llm_rubric_score_source": "llm_or_mock_output",
                "evidence_factor_name": "evidence_quality_factor * overlap_factor * llm_selection_factor",
                "evidence_factor_value": safe_float(aggregation_component["mean_evidence_quality_factor"]),
                "evidence_quality_factor": safe_float(aggregation_component["mean_evidence_quality_factor"]),
                "overlap_factor": safe_float(aggregation_component["mean_overlap_factor"]),
                "llm_selection_factor": safe_float(aggregation_component["mean_llm_selection_factor"]),
                "evidence_set_quality_factor": "",
                "source_diversity_factor": "",
                "formal_document_factor": "",
                "strategic_section_factor": "",
                "best_evidence_score_0_5": safe_float(question_row.get("best_evidence_contribution_0_5")),
                "best_distinct_year_evidence_score_0_5": "",
                "mean_top_evidence_by_year_0_5": "",
                "base_score_before_bonus_0_5": "",
                "bonus_name": "",
                "bonus_value_0_5": 0.0,
                "final_question_score_0_5": safe_float(question_row.get("question_score_0_5")),
                "final_question_score_0_100": safe_float(question_row.get("question_score_0_5")) / 5.0 * 100.0,
                "aggregation_component_json": json.dumps(aggregation_component, ensure_ascii=False),
                "needs_human_review": bool(question_row.get("needs_human_review", False)),
            }
        )

    return pd.DataFrame(rows)


def build_hc_components(hc_company: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in hc_company.iterrows():
        aggregation_component = {
            "best_evidence_score": safe_float(row.get("best_evidence_score")),
            "best_distinct_year_evidence_score": safe_float(row.get("best_distinct_year_evidence_score")),
            "mean_top_evidence_by_year": safe_float(row.get("mean_top_evidence_by_year")),
            "base_score_before_bonus": safe_float(row.get("hc_base_score_0_5")),
            "distinct_year_count": safe_float(row.get("distinct_year_count")),
            "evidence_count": safe_float(row.get("evidence_count")),
        }

        rows.append(
            {
                "company": safe_str(row.get("company")),
                "dimension": "history_consistency",
                "question_id": safe_str(row.get("hc_question_id", "HC_Q1")),
                "question_name": "History Consistency",
                "llm_rubric_score_0_5": "",
                "llm_rubric_score_source": "not_run; fallback evidence scoring used",
                "evidence_factor_name": "best_evidence + best_distinct_year_evidence + mean_top_evidence_by_year",
                "evidence_factor_value": safe_float(row.get("hc_base_score_0_5")),
                "evidence_quality_factor": "",
                "overlap_factor": "",
                "llm_selection_factor": "",
                "evidence_set_quality_factor": "",
                "source_diversity_factor": "",
                "formal_document_factor": "",
                "strategic_section_factor": "",
                "best_evidence_score_0_5": safe_float(row.get("best_evidence_score")),
                "best_distinct_year_evidence_score_0_5": safe_float(row.get("best_distinct_year_evidence_score")),
                "mean_top_evidence_by_year_0_5": safe_float(row.get("mean_top_evidence_by_year")),
                "base_score_before_bonus_0_5": safe_float(row.get("hc_base_score_0_5")),
                "bonus_name": "hc_history_bonus",
                "bonus_value_0_5": safe_float(row.get("hc_history_bonus")),
                "final_question_score_0_5": safe_float(row.get("hc_final_score_0_5")),
                "final_question_score_0_100": safe_float(row.get("hc_score_0_100")),
                "aggregation_component_json": json.dumps(aggregation_component, ensure_ascii=False),
                "needs_human_review": bool(row.get("needs_human_review", False)),
            }
        )

    return pd.DataFrame(rows)


def build_company_summary(
    pa_company: pd.DataFrame,
    sa_company: pd.DataFrame,
    hc_company: pd.DataFrame,
) -> pd.DataFrame:
    companies = sorted(
        set(pa_company.get("company", pd.Series(dtype=str)).astype(str))
        | set(sa_company.get("company", pd.Series(dtype=str)).astype(str))
        | set(hc_company.get("company", pd.Series(dtype=str)).astype(str))
    )

    pa_map = pa_company.set_index("company").to_dict(orient="index") if not pa_company.empty else {}
    sa_map = sa_company.set_index("company").to_dict(orient="index") if not sa_company.empty else {}
    hc_map = hc_company.set_index("company").to_dict(orient="index") if not hc_company.empty else {}

    rows = []
    for company in companies:
        pa = pa_map.get(company, {})
        sa = sa_map.get(company, {})
        hc = hc_map.get(company, {})
        pa_score_0_5 = safe_float(pa.get("PA_score_0_5"))
        sa_score_0_5 = safe_float(sa.get("sa_final_score_0_5"))
        hc_score_0_5 = safe_float(hc.get("hc_final_score_0_5"))
        pa_score_0_100 = safe_float(pa.get("PA_score_0_100"))
        sa_score_0_100 = safe_float(sa.get("sa_score_0_100"))
        hc_score_0_100 = safe_float(hc.get("hc_score_0_100"))
        rows.append(
            {
                "company": company,
                "purpose_articulation_score_0_5": pa_score_0_5,
                "strategy_alignment_score_0_5": sa_score_0_5,
                "history_consistency_score_0_5": hc_score_0_5,
                "aggregate_mean_score_0_5": (pa_score_0_5 + sa_score_0_5 + hc_score_0_5) / 3.0,
                "purpose_articulation_score_0_100": pa_score_0_100,
                "strategy_alignment_score_0_100": sa_score_0_100,
                "history_consistency_score_0_100": hc_score_0_100,
                "aggregate_mean_score_0_100": (
                    pa_score_0_100 + sa_score_0_100 + hc_score_0_100
                ) / 3.0,
            }
        )

    return pd.DataFrame(rows)


def build_detail(
    pa_detail: pd.DataFrame,
    sa_detail: pd.DataFrame,
    hc_detail: pd.DataFrame,
    purpose_by_company: Dict[str, str],
) -> pd.DataFrame:
    frames = []

    pa = pd.DataFrame()
    if not pa_detail.empty:
        pa = pd.DataFrame(
            {
                "company": pa_detail.get("company"),
                "dimension": "purpose_articulation",
                "question_id": pa_detail.get("question_id"),
                "chunk_id": pa_detail.get("chunk_id"),
                "source": pa_detail.get("source"),
                "year": pa_detail.get("year"),
                "section": pa_detail.get("section"),
                "evidence_text": pa_detail.get("text_clean"),
                "llm_comment": pa_detail.get("llm_reason"),
                "purpose_extract": pa_detail.get("extracted_purpose"),
                "llm_rubric_score_0_5": pa_detail.get("llm_score_0_5").combine_first(pa_detail.get("llm_set_score_0_5")),
                "evidence_quality_factor": pa_detail.get("evidence_quality_factor").combine_first(pa_detail.get("evidence_set_quality_factor")),
                "bonus_value_0_5": pa_detail.get("pa_tone_bonus"),
                "evidence_contribution_0_5": pa_detail.get("pa_evidence_contribution"),
            }
        )
        frames.append(pa)

    if not sa_detail.empty:
        frames.append(
            pd.DataFrame(
                {
                    "company": sa_detail.get("company"),
                    "dimension": "strategy_alignment",
                    "question_id": sa_detail.get("question_id"),
                    "chunk_id": sa_detail.get("chunk_id"),
                    "source": sa_detail.get("source"),
                    "year": sa_detail.get("year"),
                    "section": sa_detail.get("section"),
                    "evidence_text": sa_detail.get("text_clean"),
                    "llm_comment": sa_detail.get("llm_reasoning"),
                    "purpose_extract": sa_detail.get("extracted_purpose"),
                    "llm_rubric_score_0_5": sa_detail.get("llm_score_0_5"),
                    "evidence_quality_factor": sa_detail.get("evidence_quality_factor"),
                    "bonus_value_0_5": 0.0,
                    "evidence_contribution_0_5": sa_detail.get("evidence_contribution_0_5"),
                }
            )
        )

    if not hc_detail.empty:
        purpose = hc_detail.get("company").astype(str).map(lambda company: purpose_by_company.get(company, ""))
        frames.append(
            pd.DataFrame(
                {
                    "company": hc_detail.get("company"),
                    "dimension": "history_consistency",
                    "question_id": "HC_Q1",
                    "chunk_id": hc_detail.get("chunk_id"),
                    "source": hc_detail.get("source"),
                    "year": hc_detail.get("year"),
                    "section": hc_detail.get("section"),
                    "evidence_text": hc_detail.get("text_clean"),
                    "llm_comment": "",
                    "purpose_extract": purpose,
                    "llm_rubric_score_0_5": "",
                    "evidence_quality_factor": hc_detail.get("hc_evidence_quality_factor"),
                    "bonus_value_0_5": "",
                    "evidence_contribution_0_5": hc_detail.get("hc_evidence_contribution_0_5"),
                }
            )
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run(args: argparse.Namespace) -> Dict[str, Path]:
    pa_dir = Path(args.pa_dir)
    sa_dir = Path(args.sa_dir)
    hc_dir = Path(args.hc_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pa_company = read_csv(pa_dir / "pa_company_score_v1.csv")
    pa_question = read_csv(pa_dir / "pa_question_score_v1.csv")
    pa_detail = read_csv(pa_dir / "pa_evidence_detail_v1.csv")

    sa_company = read_csv(sa_dir / "company_sa_score_v1.csv")
    sa_question = read_csv(sa_dir / "company_sa_question_scores_v1.csv")
    sa_detail = read_csv(sa_dir / "sa_evidence_details_v1.csv")

    hc_company = read_csv(hc_dir / "hc_company_score_v1.csv")
    hc_detail = read_csv(hc_dir / "hc_evidence_details_v1.csv")

    purpose_by_company = get_company_purpose(pa_detail, sa_company)

    components = pd.concat(
        [
            build_pa_components(pa_company, pa_question, pa_detail),
            build_sa_components(sa_company, sa_question, sa_detail),
            build_hc_components(hc_company),
        ],
        ignore_index=True,
    )
    company_summary = build_company_summary(pa_company, sa_company, hc_company)
    detail = build_detail(pa_detail, sa_detail, hc_detail, purpose_by_company)

    paths = {
        "company_summary": output_dir / "aggregate_company_summary.csv",
        "summary": output_dir / "Summary.csv",
        "rubric_components": output_dir / "aggregate_rubric_components.csv",
        "detail": output_dir / "aggregate_detail.csv",
    }

    company_summary.to_csv(paths["company_summary"], index=False, encoding="utf-8-sig")
    company_summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    components.to_csv(paths["rubric_components"], index=False, encoding="utf-8-sig")
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate PA, SA, and HC test outputs into clear score components."
    )
    parser.add_argument(
        "--run-phases",
        action="store_true",
        help="Run PA, SA, and HC before aggregation with company_phase progress tracking.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_TEST_INPUT),
        help="Unified chunks CSV used when --run-phases is set.",
    )
    parser.add_argument(
        "--provider",
        default="claude",
        choices=["openai", "claude"],
        help="LLM provider used when --run-phases is set.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model used when --run-phases is set.",
    )
    parser.add_argument(
        "--phase-output-root",
        default=str(DEFAULT_PHASE_OUTPUT_ROOT),
        help="Root directory for per-company phase outputs when --run-phases is set.",
    )
    parser.add_argument(
        "--progress-path",
        default=None,
        help="Progress JSON path. Defaults to <phase-output-root>/progress_company_phase.json.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed company_phase entries in the progress JSON.",
    )
    parser.add_argument("--pa-dir", default=str(DEFAULT_PA_DIR))
    parser.add_argument("--sa-dir", default=str(DEFAULT_SA_DIR))
    parser.add_argument("--hc-dir", default=str(DEFAULT_HC_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llm_model is None:
        if args.provider == "openai":
            args.llm_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        elif args.provider == "claude":
            args.llm_model = os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)

    progress_path = None
    if args.run_phases:
        progress_path = Path(args.progress_path or Path(args.phase_output_root) / "progress_company_phase.json")
        pa_dir, sa_dir, hc_dir = run_scoring_phases(args)
        args.pa_dir = str(pa_dir)
        args.sa_dir = str(sa_dir)
        args.hc_dir = str(hc_dir)

    paths = run(args)
    if progress_path is not None:
        progress_table = write_progress_table(progress_path, Path(args.output_dir))
        if progress_table is not None:
            paths["progress"] = progress_table

    print("Aggregate outputs written:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
