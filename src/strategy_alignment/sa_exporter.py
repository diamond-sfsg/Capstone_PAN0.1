# src/strategy_alignment/sa_exporter.py

"""
Export utilities for Strategy & Source Alignment.

Exports:
1. company-level SA scores
2. question-level SA scores
3. evidence-level details
4. diagnostics report
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .sa_config import (
    PHASE_OUTPUT_DIR,
    SA_EVIDENCE_OUTPUT_PATH,
    SA_SCORE_OUTPUT_PATH,
    SA_DIAGNOSTICS_OUTPUT_PATH,
    SA_SCORE_OUTPUT_COLUMNS,
    SA_EVIDENCE_OUTPUT_COLUMNS,
    SA_QUESTION_QUERIES,
)


EVIDENCE_LIBRARY_COLUMNS = [
    "company",
    "dimension",
    "question_id",
    "question_name",
    "question_text",
    "scoring_type",
    "retrieval_stage",
    "selected_for_llm",
    "selected_for_llm_reason",
    "used_in_final_scoring",
    "used_in_question_top_evidence",
    "top_evidence_role",
    "participates_in_bonus",
    "bonus_component",
    "chunk_id",
    "doc_id",
    "source",
    "normalized_source",
    "section",
    "subsection",
    "year",
    "text_clean",
    "candidate_rank",
    "llm_input_rank",
    "evidence_set_rank",
    "keyword_relevance",
    "context_completeness",
    "rag_similarity",
    "base_evidence_score_0_1",
    "source_prior",
    "source_adjusted_rank_score",
    "evidence_quality_factor",
    "llm_score_0_5",
    "evidence_contribution_0_5",
    "llm_reasoning",
    "purpose_connection_type",
    "llm_marked_best_support",
    "llm_marked_weak_or_contradictory",
    "llm_selection_factor",
    "score_label",
]


# ============================================================
# Helpers
# ============================================================

def ensure_output_dir(output_dir: Path = PHASE_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _safe_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _reorder_columns(
    df: pd.DataFrame,
    preferred_columns: Iterable[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    preferred = [col for col in preferred_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in preferred]

    return df[preferred + remaining]


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_output_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ============================================================
# Score exports
# ============================================================

def export_company_scores(
    score_rows: List[Dict[str, Any]],
    output_path: Path = SA_SCORE_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Export company-level SA score table.
    """
    df = _safe_df(score_rows)
    df = _reorder_columns(df, SA_SCORE_OUTPUT_COLUMNS)

    _safe_to_csv(df, output_path)

    return df


def export_question_scores(
    question_rows: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Export question-level SA score table.
    """
    if output_path is None:
        output_path = PHASE_OUTPUT_DIR / "company_sa_question_scores_v1.csv"

    df = _safe_df(question_rows)

    preferred_columns = [
        "company",
        "question_id",
        "question_name",
        "question_score_0_5",
        "num_evidence",
        "best_evidence_id",
        "second_best_evidence_id",
        "best_evidence_contribution_0_5",
        "second_best_evidence_contribution_0_5",
        "needs_human_review",
        "review_reason",
    ]

    df = _reorder_columns(df, preferred_columns)

    _safe_to_csv(df, output_path)

    return df


# ============================================================
# Evidence exports
# ============================================================

def export_evidence_details(
    evidence_frames: List[pd.DataFrame],
    output_path: Path = SA_EVIDENCE_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Export evidence-level SA details.

    evidence_frames usually come from flatten_evidence_map().
    """
    if not evidence_frames:
        df = pd.DataFrame(columns=SA_EVIDENCE_OUTPUT_COLUMNS)
    else:
        valid_frames = [frame for frame in evidence_frames if not frame.empty]

        if valid_frames:
            df = pd.concat(valid_frames, ignore_index=True)
        else:
            df = pd.DataFrame(columns=SA_EVIDENCE_OUTPUT_COLUMNS)

    df = _reorder_columns(df, SA_EVIDENCE_OUTPUT_COLUMNS)

    _safe_to_csv(df, output_path)

    return df


def export_candidate_details(
    candidate_frames: List[pd.DataFrame],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Optional export: retrieved candidates before LLM scoring.
    Useful for debugging retrieval quality.
    """
    if output_path is None:
        output_path = PHASE_OUTPUT_DIR / "sa_candidate_details_v1.csv"

    if not candidate_frames:
        df = pd.DataFrame()
    else:
        valid_frames = [frame for frame in candidate_frames if not frame.empty]
        df = (
            pd.concat(valid_frames, ignore_index=True)
            if valid_frames
            else pd.DataFrame()
        )

    _safe_to_csv(df, output_path)

    return df


def normalize_sa_evidence_library(
    candidate_df: pd.DataFrame,
    evidence_df: Optional[pd.DataFrame] = None,
    question_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Normalize SA retrieval output and make selection/scoring status explicit.
    """
    if candidate_df is None or candidate_df.empty:
        return pd.DataFrame(columns=EVIDENCE_LIBRARY_COLUMNS)

    df = candidate_df.copy()
    out = pd.DataFrame()
    out["company"] = df.get("company", "")
    out["dimension"] = "strategy_alignment"
    out["question_id"] = df.get("question_id", "")
    out["question_name"] = df.get("question_name", "")
    out["question_text"] = out["question_id"].map(
        lambda value: SA_QUESTION_QUERIES.get(str(value), "")
    )
    out["scoring_type"] = "evidence"
    out["retrieval_stage"] = "llm_candidate_pack"
    out["selected_for_llm"] = True
    out["selected_for_llm_reason"] = (
        "Retrieved as candidate evidence and included in the SA question prompt."
    )
    out["used_in_final_scoring"] = False
    out["used_in_question_top_evidence"] = False
    out["top_evidence_role"] = ""
    out["participates_in_bonus"] = False
    out["bonus_component"] = ""
    out["chunk_id"] = df.get("chunk_id", "")
    out["doc_id"] = df.get("doc_id", "")
    out["source"] = df.get("source", "")
    out["normalized_source"] = df.get("normalized_source", df.get("source", ""))
    out["section"] = df.get("section", "")
    out["subsection"] = df.get("subsection", "")
    out["year"] = df.get("year", "")
    out["text_clean"] = df.get("text_clean", "")
    out["candidate_rank"] = df.get("rank", "")
    out["llm_input_rank"] = ""
    out["evidence_set_rank"] = ""
    out["keyword_relevance"] = df.get("keyword_relevance", "")
    out["context_completeness"] = df.get("context_completeness", "")
    out["rag_similarity"] = df.get("rag_similarity", "")
    out["base_evidence_score_0_1"] = df.get("base_evidence_score_0_1", "")
    out["source_prior"] = df.get("source_prior", "")
    out["source_adjusted_rank_score"] = df.get("rank_score", "")
    out["evidence_quality_factor"] = df.get("evidence_quality_factor", "")
    out["llm_score_0_5"] = ""
    out["evidence_contribution_0_5"] = ""
    out["llm_reasoning"] = ""
    out["purpose_connection_type"] = ""
    out["llm_marked_best_support"] = ""
    out["llm_marked_weak_or_contradictory"] = ""
    out["llm_selection_factor"] = ""
    out["score_label"] = ""

    key_cols = ["company", "question_id", "chunk_id"]

    if evidence_df is not None and not evidence_df.empty:
        detail_cols = [
            col
            for col in [
                *key_cols,
                "llm_score_0_5",
                "evidence_contribution_0_5",
                "llm_reasoning",
                "purpose_connection_type",
                "llm_marked_best_support",
                "llm_marked_weak_or_contradictory",
                "llm_selection_factor",
                "score_label",
            ]
            if col in evidence_df.columns
        ]

        if all(col in out.columns for col in key_cols) and all(
            col in evidence_df.columns for col in key_cols
        ):
            detail = evidence_df[detail_cols].drop_duplicates(subset=key_cols)
            out = out.merge(detail, on=key_cols, how="left", suffixes=("", "_scored"))

            for col in [
                "llm_score_0_5",
                "evidence_contribution_0_5",
                "llm_reasoning",
                "purpose_connection_type",
                "llm_marked_best_support",
                "llm_marked_weak_or_contradictory",
                "llm_selection_factor",
                "score_label",
            ]:
                scored_col = f"{col}_scored"
                if scored_col in out.columns:
                    out[col] = out[scored_col].combine_first(out[col])
                    out = out.drop(columns=[scored_col])

            out["used_in_final_scoring"] = out["evidence_contribution_0_5"].notna()

    if question_df is not None and not question_df.empty:
        role_by_key = {}
        for _, row in question_df.iterrows():
            company = str(row.get("company", ""))
            question_id = str(row.get("question_id", ""))
            best_id = str(row.get("best_evidence_id", ""))
            second_id = str(row.get("second_best_evidence_id", ""))

            if best_id:
                role_by_key[(company, question_id, best_id)] = "best_evidence"
            if second_id:
                role_by_key[(company, question_id, second_id)] = "second_best_evidence"

        out["top_evidence_role"] = out.apply(
            lambda row: role_by_key.get(
                (
                    str(row.get("company", "")),
                    str(row.get("question_id", "")),
                    str(row.get("chunk_id", "")),
                ),
                "",
            ),
            axis=1,
        )
        out["used_in_question_top_evidence"] = out["top_evidence_role"].ne("")

    return out[EVIDENCE_LIBRARY_COLUMNS]


def export_evidence_library(
    candidate_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    question_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Export the SA EvidenceLibrary with explicit LLM/scoring selection flags.
    """
    if output_path is None:
        output_path = PHASE_OUTPUT_DIR / "sa_evidence_library_v1.csv"

    df = normalize_sa_evidence_library(
        candidate_df=candidate_df,
        evidence_df=evidence_df,
        question_df=question_df,
    )

    _safe_to_csv(df, output_path)

    return df


# ============================================================
# Diagnostics
# ============================================================

def build_diagnostics_text(
    score_df: pd.DataFrame,
    question_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    candidate_df: Optional[pd.DataFrame] = None,
    output_dir: Path = PHASE_OUTPUT_DIR,
) -> str:
    """
    Build terminal / text diagnostics for SA scoring.
    """
    lines = []

    lines.append("STRATEGY & SOURCE ALIGNMENT DIAGNOSTICS")
    lines.append("=" * 80)

    lines.append("")
    lines.append("1. Company-level output")
    lines.append("-" * 80)
    lines.append(f"companies_scored: {len(score_df)}")

    if not score_df.empty:
        lines.append(
            f"mean_sa_final_score_0_5: "
            f"{score_df['sa_final_score_0_5'].mean():.4f}"
        )
        lines.append(
            f"mean_sa_score_0_100: "
            f"{score_df['sa_score_0_100'].mean():.4f}"
        )
        lines.append(
            f"needs_human_review_count: "
            f"{int(score_df['sa_needs_human_review'].sum())}"
        )
        lines.append(
            f"needs_human_review_rate: "
            f"{score_df['sa_needs_human_review'].mean():.4f}"
        )

    lines.append("")
    lines.append("2. Question-level output")
    lines.append("-" * 80)
    lines.append(f"question_rows: {len(question_df)}")

    if not question_df.empty:
        q_summary = (
            question_df.groupby("question_id")
            .agg(
                count=("company", "count"),
                mean_score=("question_score_0_5", "mean"),
                review_rate=("needs_human_review", "mean"),
                mean_num_evidence=("num_evidence", "mean"),
            )
            .reset_index()
        )

        for _, row in q_summary.iterrows():
            lines.append(
                f"{row['question_id']}: "
                f"count={int(row['count'])}, "
                f"mean_score={row['mean_score']:.4f}, "
                f"review_rate={row['review_rate']:.4f}, "
                f"mean_num_evidence={row['mean_num_evidence']:.2f}"
            )

    lines.append("")
    lines.append("3. Evidence-level output")
    lines.append("-" * 80)
    lines.append(f"evidence_rows: {len(evidence_df)}")

    if not evidence_df.empty:
        if "question_id" in evidence_df.columns:
            evidence_counts = (
                evidence_df.groupby("question_id")
                .size()
                .reset_index(name="evidence_count")
            )

            for _, row in evidence_counts.iterrows():
                lines.append(
                    f"{row['question_id']}: evidence_count={int(row['evidence_count'])}"
                )

        if "source" in evidence_df.columns:
            lines.append("")
            lines.append("Evidence by source:")
            source_counts = (
                evidence_df.groupby("source")
                .size()
                .sort_values(ascending=False)
                .reset_index(name="count")
            )

            for _, row in source_counts.iterrows():
                lines.append(f"{row['source']}: {int(row['count'])}")

    if candidate_df is not None:
        lines.append("")
        lines.append("4. Candidate retrieval output")
        lines.append("-" * 80)
        lines.append(f"candidate_rows: {len(candidate_df)}")

        if not candidate_df.empty and "question_id" in candidate_df.columns:
            candidate_counts = (
                candidate_df.groupby("question_id")
                .size()
                .reset_index(name="candidate_count")
            )

            for _, row in candidate_counts.iterrows():
                lines.append(
                    f"{row['question_id']}: candidate_count={int(row['candidate_count'])}"
                )

    lines.append("")
    lines.append("Output files")
    lines.append("-" * 80)
    lines.append(f"company_score_file: {output_dir / SA_SCORE_OUTPUT_PATH.name}")
    lines.append(f"evidence_detail_file: {output_dir / SA_EVIDENCE_OUTPUT_PATH.name}")
    lines.append(f"evidence_library_file: {output_dir / 'sa_evidence_library_v1.csv'}")
    lines.append(f"diagnostics_file: {output_dir / SA_DIAGNOSTICS_OUTPUT_PATH.name}")

    return "\n".join(lines)


def export_diagnostics(
    score_df: pd.DataFrame,
    question_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    candidate_df: Optional[pd.DataFrame] = None,
    output_path: Path = SA_DIAGNOSTICS_OUTPUT_PATH,
    output_dir: Path = PHASE_OUTPUT_DIR,
    print_to_console: bool = True,
) -> str:
    """
    Export diagnostics report as txt.
    """
    text = build_diagnostics_text(
        score_df=score_df,
        question_df=question_df,
        evidence_df=evidence_df,
        candidate_df=candidate_df,
        output_dir=output_dir,
    )

    ensure_output_dir(output_path.parent)
    output_path.write_text(text, encoding="utf-8")

    if print_to_console:
        print(text)

    return text


# ============================================================
# Unified export wrapper
# ============================================================

def export_sa_outputs(
    score_rows: List[Dict[str, Any]],
    question_rows: List[Dict[str, Any]],
    evidence_frames: List[pd.DataFrame],
    candidate_frames: Optional[List[pd.DataFrame]] = None,
    output_dir: Path = PHASE_OUTPUT_DIR,
    print_diagnostics: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Export all SA outputs.
    """
    output_dir = Path(output_dir)
    ensure_output_dir(output_dir)

    score_df = export_company_scores(score_rows, output_dir / SA_SCORE_OUTPUT_PATH.name)
    question_df = export_question_scores(
        question_rows,
        output_dir / "company_sa_question_scores_v1.csv",
    )
    evidence_df = export_evidence_details(
        evidence_frames,
        output_dir / SA_EVIDENCE_OUTPUT_PATH.name,
    )

    if candidate_frames is not None:
        candidate_df = export_candidate_details(
            candidate_frames,
            output_dir / "sa_candidate_details_v1.csv",
        )
        evidence_library_df = export_evidence_library(
            candidate_df=candidate_df,
            evidence_df=evidence_df,
            question_df=question_df,
            output_path=output_dir / "sa_evidence_library_v1.csv",
        )
    else:
        candidate_df = None
        evidence_library_df = None

    export_diagnostics(
        score_df=score_df,
        question_df=question_df,
        evidence_df=evidence_df,
        candidate_df=candidate_df,
        output_path=output_dir / SA_DIAGNOSTICS_OUTPUT_PATH.name,
        output_dir=output_dir,
        print_to_console=print_diagnostics,
    )

    return {
        "score_df": score_df,
        "question_df": question_df,
        "evidence_df": evidence_df,
        "candidate_df": candidate_df,
        "evidence_library_df": evidence_library_df,
    }
