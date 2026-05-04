from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# PATH SETUP
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from evidence_llm_review_v1.llm_review_config import (
    PHASE2_INPUT_DIR,
    PHASE3_OUTPUT_DIR,
    LLM_PROVIDER,
    LLM_MODEL_ENV_VAR,
    LLM_API_KEY_ENV_VAR,
    RUN_VERSION,
    EVIDENCE_POOL_CONFIG,
    OUTPUT_FILES,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    MAX_ROWS_PER_BUCKET,
    SAVE_EVERY_N_ROWS,
    ID_COL,
    TEXT_COL_CANDIDATES,
    REQUIRED_BASE_COLS,
    OPTIONAL_CONTEXT_COLS,
    COMMON_LLM_REVIEW_FIELDS,
    PA_ONLY_LLM_FIELDS,
    ALL_LLM_REVIEW_FIELDS,
    HUMAN_REVIEW_RULES,
    DEFAULT_LLM_MODEL,
)

from evidence_llm_review_v1.prompt_templates import (
    SYSTEM_PROMPT,
    build_review_prompt,
    expected_fields_for_bucket,
    empty_review_result,
)


# =============================================================================
# ENV / CLIENT
# =============================================================================

def load_env_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.getenv(LLM_API_KEY_ENV_VAR):
        try:
            from configs.config import OPENAI_API_KEY
        except ImportError:
            return

        if OPENAI_API_KEY:
            os.environ[LLM_API_KEY_ENV_VAR] = OPENAI_API_KEY


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI SDK is not installed. Run: pip install openai python-dotenv"
        ) from exc

    api_key = os.getenv(LLM_API_KEY_ENV_VAR)

    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Please set environment variable: {LLM_API_KEY_ENV_VAR}"
        )

    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    model = os.getenv(LLM_MODEL_ENV_VAR)

    if not model:
        return DEFAULT_LLM_MODEL

    return model


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_input_df(df: pd.DataFrame, bucket_name: str) -> None:
    missing_base = [col for col in REQUIRED_BASE_COLS if col not in df.columns]

    if missing_base:
        raise ValueError(
            f"{bucket_name}: missing required columns: {missing_base}"
        )

    if ID_COL not in df.columns:
        raise ValueError(f"{bucket_name}: missing ID column: {ID_COL}")

    text_cols_found = [col for col in TEXT_COL_CANDIDATES if col in df.columns]
    if not text_cols_found:
        raise ValueError(
            f"{bucket_name}: no text column found. Expected one of: {TEXT_COL_CANDIDATES}"
        )

    duplicated = df[ID_COL].duplicated().sum()
    if duplicated:
        print(f"[WARN] {bucket_name}: {duplicated:,} duplicated {ID_COL} values.")


def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "y"}:
            return True
        if lowered in {"false", "no", "0", "n"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return True


def clamp_float(value: Any, min_value: float, max_value: float, default: float | None = None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return max(min_value, min(max_value, result))


def clamp_score(value: Any):
    result = clamp_float(value, 0, 5, default=None)
    if result is None:
        return None
    return int(round(result))


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Robust JSON extraction.

    JSON mode usually returns clean JSON, but this fallback protects against
    accidental surrounding text.
    """
    if not text:
        raise ValueError("Empty model response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:300]}")

    candidate = text[start : end + 1]
    return json.loads(candidate)


def normalize_review_result(
    raw_result: dict[str, Any],
    bucket_name: str,
) -> dict[str, Any]:
    """
    Make sure every expected field exists and has a usable type.
    """
    expected_fields = expected_fields_for_bucket(bucket_name)
    fallback = empty_review_result(bucket_name)

    normalized = {}

    for field in expected_fields:
        value = raw_result.get(field, fallback.get(field))

        if field == "llm_needs_human_review":
            normalized[field] = normalize_bool(value)

        elif field == "llm_confidence":
            normalized[field] = clamp_float(
                value,
                min_value=0.0,
                max_value=1.0,
                default=0.0,
            )

        elif field == "llm_review_reason":
            reason = "" if value is None else str(value).strip()
            normalized[field] = reason[:500]

        else:
            normalized[field] = clamp_score(value)

    # For non-PA buckets, keep PA-only fields as None so all outputs can merge cleanly.
    if bucket_name != "purpose_articulation":
        for field in PA_ONLY_LLM_FIELDS:
            normalized[field] = None

    return normalized


# =============================================================================
# OPENAI CALL
# =============================================================================

def call_openai_review(
    client,
    model: str,
    prompt: str,
    bucket_name: str,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> tuple[dict[str, Any], str, str]:
    """
    Returns:
        normalized_result, raw_response_text, error_message
    """
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content or ""
            parsed = extract_json_object(raw_text)
            normalized = normalize_review_result(parsed, bucket_name=bucket_name)

            return normalized, raw_text, ""

        except Exception as exc:
            last_error = str(exc)

            # Some newer models/API variants may prefer max_completion_tokens.
            if attempt == 1 and "max_tokens" in last_error:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_completion_tokens=MAX_OUTPUT_TOKENS,
                        response_format={"type": "json_object"},
                    )

                    raw_text = response.choices[0].message.content or ""
                    parsed = extract_json_object(raw_text)
                    normalized = normalize_review_result(parsed, bucket_name=bucket_name)

                    return normalized, raw_text, ""

                except Exception as retry_exc:
                    last_error = str(retry_exc)

            if attempt < max_retries:
                wait_time = retry_sleep * attempt
                print(
                    f"[WARN] OpenAI call failed for {bucket_name}. "
                    f"Attempt {attempt}/{max_retries}. Retrying in {wait_time:.1f}s."
                )
                time.sleep(wait_time)

    fallback = empty_review_result(bucket_name)
    normalized = normalize_review_result(fallback, bucket_name=bucket_name)

    return normalized, "", last_error


# =============================================================================
# HUMAN REVIEW LOGIC
# =============================================================================

def compute_human_review_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    low_relevance_threshold = HUMAN_REVIEW_RULES["low_bucket_relevance_threshold"]
    low_credibility_threshold = HUMAN_REVIEW_RULES["low_credibility_threshold"]
    high_boilerplate_threshold = HUMAN_REVIEW_RULES["high_boilerplate_risk_threshold"]
    problematic_flags = HUMAN_REVIEW_RULES["problematic_phase2_review_flags"]

    df["flag_low_llm_relevance"] = (
        df["llm_bucket_relevance_score"].fillna(0) <= low_relevance_threshold
    )

    df["flag_low_llm_credibility"] = (
        df["llm_credibility_score"].fillna(0) <= low_credibility_threshold
    )

    df["flag_high_boilerplate_risk"] = (
        df["llm_boilerplate_risk_score"].fillna(0) >= high_boilerplate_threshold
    )

    if "review_flag" in df.columns:
        df["flag_phase2_review_issue"] = df["review_flag"].isin(problematic_flags)
    else:
        df["flag_phase2_review_issue"] = False

    df["flag_llm_error"] = df["llm_review_status"] != "ok"

    df["final_needs_human_review"] = (
        df["llm_needs_human_review"].fillna(True)
        | df["flag_low_llm_relevance"]
        | df["flag_low_llm_credibility"]
        | df["flag_high_boilerplate_risk"]
        | df["flag_phase2_review_issue"]
        | df["flag_llm_error"]
    )

    def make_reason(row) -> str:
        reasons = []

        if row.get("llm_needs_human_review"):
            reasons.append("llm_marked_review")
        if row.get("flag_low_llm_relevance"):
            reasons.append("low_relevance")
        if row.get("flag_low_llm_credibility"):
            reasons.append("low_credibility")
        if row.get("flag_high_boilerplate_risk"):
            reasons.append("high_boilerplate")
        if row.get("flag_phase2_review_issue"):
            reasons.append("phase2_overlap_or_margin")
        if row.get("flag_llm_error"):
            reasons.append("llm_error")

        return ";".join(reasons) if reasons else "ok"

    df["human_review_reason_code"] = df.apply(make_reason, axis=1)

    return df


# =============================================================================
# PROCESS ONE BUCKET
# =============================================================================

def input_path_for_bucket(bucket_name: str) -> Path:
    return PHASE2_INPUT_DIR / EVIDENCE_POOL_CONFIG[bucket_name]["input_csv"]


def output_path_for_bucket(bucket_name: str) -> Path:
    return PHASE3_OUTPUT_DIR / OUTPUT_FILES[bucket_name]


def assert_output_writable(output_path: Path) -> None:
    if not output_path.exists():
        return

    try:
        with output_path.open("a", encoding="utf-8", newline=""):
            pass
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to output file because it is locked: {output_path}\n"
            "Close Excel/CSV preview windows for this file, or pause OneDrive sync, then rerun."
        ) from exc


def process_bucket(
    bucket_name: str,
    client,
    model: str,
    limit: int | None = None,
    force: bool = False,
    resume: bool = False,
) -> pd.DataFrame:
    input_path = input_path_for_bucket(bucket_name)
    output_path = output_path_for_bucket(bucket_name)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found for {bucket_name}: {input_path}")

    assert_output_writable(output_path)

    if output_path.exists() and not force and not resume:
        print(f"[SKIP] Existing output found for {bucket_name}: {output_path}")
        print("       Use --force to overwrite.")
        return pd.read_csv(output_path)

    print(f"\n[LOAD] {bucket_name}: {input_path}")
    df = pd.read_csv(input_path)

    validate_input_df(df, bucket_name=bucket_name)

    effective_limit = limit if limit is not None else MAX_ROWS_PER_BUCKET

    if effective_limit is not None:
        df = df.head(effective_limit).copy()
        print(f"[INFO] {bucket_name}: limited to {len(df):,} rows.")

    existing_df = pd.DataFrame()
    completed_ids = set()

    if output_path.exists() and resume and not force:
        existing_df = pd.read_csv(output_path)

        if ID_COL not in existing_df.columns:
            raise ValueError(
                f"{bucket_name}: cannot resume because existing output is missing {ID_COL}: "
                f"{output_path}"
            )

        current_ids = set(df[ID_COL])
        existing_df = existing_df[existing_df[ID_COL].isin(current_ids)].copy()
        existing_df = existing_df.drop_duplicates(subset=[ID_COL], keep="last")
        completed_ids = set(existing_df[ID_COL])
        df = df[~df[ID_COL].isin(completed_ids)].copy()

        print(
            f"[RESUME] {bucket_name}: found {len(completed_ids):,} existing rows; "
            f"{len(df):,} rows remaining."
        )

        if df.empty:
            result_df = compute_human_review_flags(existing_df)
            result_df.to_csv(output_path, index=False)
            print(f"[DONE] {bucket_name}: output already complete.")
            return result_df

    results = existing_df.to_dict("records") if not existing_df.empty else []
    total = len(df)
    target_total = total + len(completed_ids)

    for idx, (_, row) in enumerate(df.iterrows(), start=len(completed_ids) + 1):
        row_dict = row.to_dict()
        prompt = build_review_prompt(row=row_dict, bucket_name=bucket_name)

        review_result, raw_response, error_message = call_openai_review(
            client=client,
            model=model,
            prompt=prompt,
            bucket_name=bucket_name,
        )

        output_row = dict(row_dict)
        output_row.update(review_result)

        output_row["llm_provider"] = LLM_PROVIDER
        output_row["llm_model"] = model
        output_row["llm_review_run_version"] = RUN_VERSION
        output_row["llm_review_timestamp"] = datetime.now().isoformat(timespec="seconds")
        output_row["llm_review_status"] = "ok" if not error_message else "error"
        output_row["llm_error_message"] = error_message
        output_row["llm_raw_response"] = raw_response

        results.append(output_row)

        if idx % 10 == 0 or idx == target_total:
            print(f"[PROGRESS] {bucket_name}: {idx:,}/{target_total:,}")

        if SAVE_EVERY_N_ROWS and idx % SAVE_EVERY_N_ROWS == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df = compute_human_review_flags(checkpoint_df)
            PHASE3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_df.to_csv(output_path, index=False)
            print(f"[CHECKPOINT] saved {len(checkpoint_df):,} rows -> {output_path}")

    result_df = pd.DataFrame(results)
    result_df = compute_human_review_flags(result_df)

    PHASE3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"[EXPORT] {bucket_name}: {len(result_df):,} rows")
    print(f"         {output_path}")

    return result_df


# =============================================================================
# MERGE / EXPORT
# =============================================================================

def export_all_reviews(bucket_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(bucket_dfs.values(), ignore_index=True)

    all_path = PHASE3_OUTPUT_DIR / OUTPUT_FILES["all_reviews"]
    all_df.to_csv(all_path, index=False)

    print(f"\n[EXPORT] all reviews: {len(all_df):,} rows")
    print(f"         {all_path}")

    return all_df


def export_human_review_queue(all_df: pd.DataFrame) -> pd.DataFrame:
    queue_df = all_df[all_df["final_needs_human_review"]].copy()

    sort_cols = [
        "llm_review_status",
        "human_review_reason_code",
        "llm_bucket_relevance_score",
        "llm_credibility_score",
        "llm_boilerplate_risk_score",
    ]

    available_sort_cols = safe_cols(queue_df, sort_cols)

    if available_sort_cols:
        queue_df = queue_df.sort_values(
            by=available_sort_cols,
            ascending=[True, True, True, True, False][: len(available_sort_cols)],
        )

    queue_path = PHASE3_OUTPUT_DIR / OUTPUT_FILES["human_review_queue"]
    queue_df.to_csv(queue_path, index=False)

    print(f"[EXPORT] human review queue: {len(queue_df):,} rows")
    print(f"         {queue_path}")

    return queue_df


def write_diagnostics(all_df: pd.DataFrame, queue_df: pd.DataFrame) -> None:
    lines = []

    lines.append("EVIDENCE LLM REVIEW V1 DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"provider    : {LLM_PROVIDER}")
    lines.append(f"run_version : {RUN_VERSION}")
    lines.append(f"rows        : {len(all_df):,}")
    lines.append(f"review rows : {len(queue_df):,}")
    lines.append(
        f"review rate : {len(queue_df) / len(all_df):.4f}"
        if len(all_df)
        else "review rate : 0.0000"
    )
    lines.append("")

    lines.append("ROWS BY BUCKET")
    lines.append("=" * 80)

    if "evidence_bucket" in all_df.columns:
        for bucket, count in all_df["evidence_bucket"].value_counts().items():
            lines.append(f"{bucket}: {count:,}")

    lines.append("")

    lines.append("LLM STATUS")
    lines.append("=" * 80)

    if "llm_review_status" in all_df.columns:
        for status, count in all_df["llm_review_status"].value_counts().items():
            lines.append(f"{status}: {count:,}")

    lines.append("")

    lines.append("COMMON SCORE SUMMARY")
    lines.append("=" * 80)

    score_cols = [
        "llm_bucket_relevance_score",
        "llm_evidence_specificity_score",
        "llm_source_context_score",
        "llm_boilerplate_risk_score",
        "llm_credibility_score",
        "llm_confidence",
    ]

    for col in score_cols:
        if col in all_df.columns:
            series = pd.to_numeric(all_df[col], errors="coerce")
            lines.append(
                f"{col}: "
                f"mean={series.mean():.4f}, "
                f"median={series.median():.4f}, "
                f"min={series.min():.4f}, "
                f"max={series.max():.4f}"
            )

    lines.append("")

    lines.append("PA-ONLY SCORE SUMMARY")
    lines.append("=" * 80)

    pa_df = all_df[all_df["evidence_bucket"] == "purpose_articulation"].copy()

    for col in PA_ONLY_LLM_FIELDS:
        if col in pa_df.columns:
            series = pd.to_numeric(pa_df[col], errors="coerce")
            lines.append(
                f"{col}: "
                f"mean={series.mean():.4f}, "
                f"median={series.median():.4f}, "
                f"min={series.min():.4f}, "
                f"max={series.max():.4f}"
            )

    lines.append("")

    lines.append("HUMAN REVIEW REASON CODES")
    lines.append("=" * 80)

    if "human_review_reason_code" in all_df.columns:
        for reason, count in all_df["human_review_reason_code"].value_counts().items():
            lines.append(f"{reason}: {count:,}")

    lines.append("")

    lines.append("ERROR SAMPLE")
    lines.append("=" * 80)

    error_df = all_df[all_df.get("llm_review_status", "") == "error"].copy()

    if error_df.empty:
        lines.append("No LLM errors.")
    else:
        for _, row in error_df.head(10).iterrows():
            lines.append(
                f"- chunk_id={row.get('chunk_id')} | "
                f"bucket={row.get('evidence_bucket')} | "
                f"error={row.get('llm_error_message')}"
            )

    diagnostics_path = PHASE3_OUTPUT_DIR / OUTPUT_FILES["diagnostics"]
    diagnostics_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[EXPORT] diagnostics: {diagnostics_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 OpenAI LLM evidence review."
    )

    parser.add_argument(
        "--bucket",
        default="all",
        choices=["all"] + list(EVIDENCE_POOL_CONFIG.keys()),
        help="Which evidence bucket to process.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit per bucket for testing.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing output files by skipping already reviewed chunk_id values.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_env_if_available()

    PHASE3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = get_openai_client()
    model = get_model_name()

    print("[START] Phase 3 LLM evidence review")
    print(f"provider    : {LLM_PROVIDER}")
    print(f"model       : {model}")
    print(f"run_version : {RUN_VERSION}")
    print(f"output_dir  : {PHASE3_OUTPUT_DIR}")

    if args.bucket == "all":
        buckets_to_run = list(EVIDENCE_POOL_CONFIG.keys())
    else:
        buckets_to_run = [args.bucket]

    bucket_dfs = {}

    for bucket_name in buckets_to_run:
        bucket_df = process_bucket(
            bucket_name=bucket_name,
            client=client,
            model=model,
            limit=args.limit,
            force=args.force,
            resume=args.resume,
        )

        bucket_dfs[bucket_name] = bucket_df

    for bucket_name in EVIDENCE_POOL_CONFIG:
        if bucket_name in bucket_dfs:
            continue

        existing_output_path = output_path_for_bucket(bucket_name)
        if existing_output_path.exists():
            bucket_dfs[bucket_name] = pd.read_csv(existing_output_path)

    all_df = export_all_reviews(bucket_dfs)
    queue_df = export_human_review_queue(all_df)
    write_diagnostics(all_df, queue_df)

    print("\nDONE: Phase 3 OpenAI LLM evidence review complete.")


if __name__ == "__main__":
    main()
