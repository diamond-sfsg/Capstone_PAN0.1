# src/strategy_alignment/sa_loader.py

"""
Data loading utilities for Strategy & Source Alignment.

This module loads:
1. unified chunk corpus
2. PA-extracted purpose reference
3. company-level SA targets

SA depends on PA because SA must compare strategy/operation evidence against
the company's stated purpose.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .sa_config import (
    INPUT_CHUNKS_PATH,
    PA_PURPOSE_REFERENCE_PATH,
    REQUIRED_CHUNK_COLUMNS,
    OPTIONAL_CHUNK_COLUMNS,
    REQUIRED_PURPOSE_REFERENCE_COLUMNS,
    OPTIONAL_PURPOSE_REFERENCE_COLUMNS,
    MIN_PURPOSE_CONFIDENCE_FOR_AUTO_SCORE,
)


# ============================================================
# Basic helpers
# ============================================================

def _ensure_path_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _ensure_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _safe_string(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item).strip() for item in value if str(item).strip())
    if pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _parse_list_like(value: Any) -> List[str]:
    """
    Parse list-like values from CSV cells.

    Handles:
    - actual list
    - JSON list string
    - Python literal list string
    - semicolon/comma-separated string
    """
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]

    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]

    return [text]


def normalize_source(source: Any) -> str:
    value = _safe_string(source).lower()

    if "edgar" in value or "10-k" in value or "10k" in value or "sec" in value:
        return "edgar"
    if "official" in value or "web" in value or "website" in value:
        return "official_web"
    if "linkedin" in value:
        return "linkedin"

    return value if value else "unknown"


def normalize_section(section: Any) -> str:
    value = _safe_string(section).lower()
    return value if value else "unknown"


def normalize_company(company: Any) -> str:
    return _safe_string(company)


# ============================================================
# Chunk loading
# ============================================================

def load_chunks(
    path: Path = INPUT_CHUNKS_PATH,
    filter_bad_quality: bool = True,
) -> pd.DataFrame:
    """
    Load unified chunks for SA scoring.

    Required columns:
    - chunk_id
    - company
    - year
    - source
    - section
    - text_clean

    Optional quality filtering removes clearly unusable chunks but does not
    remove duplicates, because duplicate / recurrence information may still
    be needed for overlap factors.
    """
    _ensure_path_exists(path, "Input chunk file")

    df = pd.read_csv(path)
    _ensure_required_columns(df, REQUIRED_CHUNK_COLUMNS, "Unified chunks")

    keep_columns = [
        col
        for col in REQUIRED_CHUNK_COLUMNS + OPTIONAL_CHUNK_COLUMNS
        if col in df.columns
    ]
    df = df[keep_columns].copy()

    df["company"] = df["company"].map(normalize_company)
    df["source"] = df["source"].map(normalize_source)
    df["section"] = df["section"].map(normalize_section)
    df["text_clean"] = df["text_clean"].fillna("").astype(str).str.strip()

    if "subsection" in df.columns:
        df["subsection"] = df["subsection"].fillna("").astype(str).str.strip()

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if filter_bad_quality and "quality_flag" in df.columns:
        bad_flags = {
            "garbled_text",
            "too_short",
            "empty",
            "invalid",
        }
        df = df[~df["quality_flag"].fillna("").str.lower().isin(bad_flags)].copy()

    df = df[df["company"].ne("")].copy()
    df = df[df["text_clean"].ne("")].copy()

    return df.reset_index(drop=True)


# ============================================================
# PA purpose reference loading
# ============================================================

def load_purpose_reference(
    path: Path = PA_PURPOSE_REFERENCE_PATH,
) -> pd.DataFrame:
    """
    Load PA-extracted purpose reference.

    This file should be produced by the PA pipeline.

    Required:
    - company
    - purpose_statement_normalized
    - purpose_confidence_0_1

    SA uses this reference to compare strategy evidence against the company's
    stated purpose.
    """
    _ensure_path_exists(path, "PA purpose reference file")

    df = pd.read_csv(path)

    if "extracted_purpose" in df.columns and "purpose_statement_normalized" not in df.columns:
        df = build_purpose_reference_from_pa_evidence_detail(df)

    _ensure_required_columns(df, REQUIRED_PURPOSE_REFERENCE_COLUMNS, "PA purpose reference")

    keep_columns = [
        col
        for col in REQUIRED_PURPOSE_REFERENCE_COLUMNS + OPTIONAL_PURPOSE_REFERENCE_COLUMNS
        if col in df.columns
    ]
    df = df[keep_columns].copy()

    df["company"] = df["company"].map(normalize_company)
    df["purpose_statement_normalized"] = (
        df["purpose_statement_normalized"].fillna("").astype(str).str.strip()
    )

    df["purpose_confidence_0_1"] = (
        df["purpose_confidence_0_1"]
        .map(lambda x: _clip(_safe_float(x, default=0.0), 0.0, 1.0))
    )

    for col in [
        "extracted_purpose",
        "purpose_statement_raw",
        "reason_for_existence",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    for col in [
        "served_stakeholders",
        "intended_impact",
        "supporting_evidence_ids",
    ]:
        if col in df.columns:
            df[col] = df[col].map(_parse_list_like)

    df = df[df["company"].ne("")].copy()
    df = df[df["purpose_statement_normalized"].ne("")].copy()

    # If multiple PA rows exist per company, keep the highest confidence row.
    df = (
        df.sort_values(["company", "purpose_confidence_0_1"], ascending=[True, False])
        .drop_duplicates(subset=["company"], keep="first")
        .reset_index(drop=True)
    )

    return df


def _support_level_rank(value: Any) -> int:
    order = {
        "none": 0,
        "weak": 1,
        "moderate": 2,
        "strong": 3,
        "very_strong": 4,
    }
    return order.get(_safe_string(value).lower(), 0)


def build_purpose_reference_from_pa_evidence_detail(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive SA purpose references from PA evidence-detail output.

    The PA pipeline now keeps extracted_purpose at evidence level. SA needs a
    company-level reference sentence, so this keeps the highest-confidence
    extracted purpose per company, preferring PA_Q1/PA_Q2 evidence rows.
    """
    required = ["company", "extracted_purpose"]
    _ensure_required_columns(pa_df, required, "PA evidence detail")

    df = pa_df.copy()
    df["company"] = df["company"].map(normalize_company)
    df["extracted_purpose"] = df["extracted_purpose"].fillna("").astype(str).str.strip()
    df = df[(df["company"] != "") & (df["extracted_purpose"] != "")].copy()

    if df.empty:
        return pd.DataFrame(columns=REQUIRED_PURPOSE_REFERENCE_COLUMNS + OPTIONAL_PURPOSE_REFERENCE_COLUMNS)

    if "question_id" not in df.columns:
        df["question_id"] = ""

    df["question_priority"] = df["question_id"].map(
        {"PA_Q1": 3, "PA_Q2": 2, "PA_Q3": 1}
    ).fillna(0)

    if "llm_score_0_5" in df.columns:
        df["purpose_score"] = pd.to_numeric(df["llm_score_0_5"], errors="coerce").fillna(0.0)
    elif "llm_set_score_0_5" in df.columns:
        df["purpose_score"] = pd.to_numeric(df["llm_set_score_0_5"], errors="coerce").fillna(0.0)
    else:
        df["purpose_score"] = 0.0

    if "support_level" in df.columns:
        df["support_rank"] = df["support_level"].map(_support_level_rank)
    else:
        df["support_rank"] = 0

    if "pa_evidence_contribution" in df.columns:
        df["purpose_contribution"] = pd.to_numeric(
            df["pa_evidence_contribution"], errors="coerce"
        ).fillna(0.0)
    else:
        df["purpose_contribution"] = 0.0

    df["purpose_confidence_0_1"] = (
        df["purpose_score"].map(lambda x: _clip(_safe_float(x, 0.0) / 5.0, 0.0, 1.0))
    )

    df = df.sort_values(
        [
            "company",
            "question_priority",
            "support_rank",
            "purpose_score",
            "purpose_contribution",
        ],
        ascending=[True, False, False, False, False],
    )

    best = df.drop_duplicates(subset=["company"], keep="first").copy()

    records = []
    for _, row in best.iterrows():
        purpose = _safe_string(row.get("extracted_purpose", ""))
        chunk_id = _safe_string(row.get("chunk_id", ""))
        records.append(
            {
                "company": _safe_string(row.get("company", "")),
                "extracted_purpose": purpose,
                "purpose_statement_normalized": purpose,
                "purpose_statement_raw": purpose,
                "purpose_confidence_0_1": _clip(
                    _safe_float(row.get("purpose_confidence_0_1", 0.0)),
                    0.0,
                    1.0,
                ),
                "supporting_evidence_ids": [chunk_id] if chunk_id else [],
                "served_stakeholders": [],
                "intended_impact": [],
                "reason_for_existence": "",
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Company-level target construction
# ============================================================

def build_company_targets(
    chunks_df: pd.DataFrame,
    purpose_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build company-level SA targets.

    A company is scorable only if:
    - it has chunk evidence
    - it has PA-extracted purpose reference

    If PA confidence is low, the target remains included but will be flagged
    for human review.
    """
    chunk_companies = set(chunks_df["company"].dropna().astype(str))
    purpose_companies = set(purpose_df["company"].dropna().astype(str))

    target_companies = sorted(chunk_companies.intersection(purpose_companies))

    targets = pd.DataFrame({"company": target_companies})
    targets = targets.merge(purpose_df, on="company", how="left")

    targets["purpose_reference_available"] = targets[
        "purpose_statement_normalized"
    ].fillna("").ne("")

    targets["low_purpose_confidence"] = (
        targets["purpose_confidence_0_1"].fillna(0.0)
        < MIN_PURPOSE_CONFIDENCE_FOR_AUTO_SCORE
    )

    targets["sa_needs_human_review"] = (
        ~targets["purpose_reference_available"]
        | targets["low_purpose_confidence"]
    )

    targets["sa_review_reason"] = targets.apply(
        _build_target_review_reason,
        axis=1,
    )

    return targets.reset_index(drop=True)


def _build_target_review_reason(row: pd.Series) -> str:
    reasons = []

    if not bool(row.get("purpose_reference_available", False)):
        reasons.append("missing_purpose_reference")

    if bool(row.get("low_purpose_confidence", False)):
        reasons.append("low_purpose_confidence")

    return ";".join(reasons)


# ============================================================
# Evidence pool utilities
# ============================================================

def get_company_chunks(
    chunks_df: pd.DataFrame,
    company: str,
) -> pd.DataFrame:
    """
    Return all evidence chunks for one company.
    """
    company_norm = normalize_company(company)
    out = chunks_df[chunks_df["company"] == company_norm].copy()

    if "year" in out.columns:
        out = out.sort_values(["year", "source", "section", "chunk_id"])

    return out.reset_index(drop=True)


def get_company_purpose_reference(
    purpose_df: pd.DataFrame,
    company: str,
) -> Dict[str, Any]:
    """
    Return PA purpose reference for one company as a dictionary.
    """
    company_norm = normalize_company(company)
    match = purpose_df[purpose_df["company"] == company_norm]

    if match.empty:
        return {
            "company": company_norm,
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


def build_company_sa_input(
    chunks_df: pd.DataFrame,
    purpose_df: pd.DataFrame,
    company: str,
) -> Dict[str, Any]:
    """
    Build the full SA input object for one company.
    """
    company_chunks = get_company_chunks(chunks_df, company)
    purpose_reference = get_company_purpose_reference(purpose_df, company)

    return {
        "company": normalize_company(company),
        "chunks": company_chunks,
        "purpose_reference": purpose_reference,
        "num_chunks": len(company_chunks),
        "num_years": (
            int(company_chunks["year"].nunique())
            if "year" in company_chunks.columns
            else None
        ),
    }


def load_all_sa_inputs(
    chunks_path: Path = INPUT_CHUNKS_PATH,
    purpose_reference_path: Path = PA_PURPOSE_REFERENCE_PATH,
) -> Dict[str, Any]:
    """
    Convenience loader for the SA pipeline.

    Returns:
    - chunks_df
    - purpose_df
    - targets_df
    """
    chunks_df = load_chunks(chunks_path)
    purpose_df = load_purpose_reference(purpose_reference_path)
    targets_df = build_company_targets(chunks_df, purpose_df)

    return {
        "chunks_df": chunks_df,
        "purpose_df": purpose_df,
        "targets_df": targets_df,
    }
