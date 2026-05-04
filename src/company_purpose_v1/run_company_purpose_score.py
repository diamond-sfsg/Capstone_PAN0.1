import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from company_purpose_v1.purpose_score_config import (
    PHASE3_INPUT_FILE,
    PHASE4_OUTPUT_DIR,
    OUTPUT_FILES,
    RUN_VERSION,
    LLM_PROVIDER,
    LLM_MODEL_ENV_VAR,
    LLM_API_KEY_ENV_VAR,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    COMPANY_COL,
    YEAR_COL,
    BUCKET_COL,
    TEXT_COL_CANDIDATES,
    ID_COL,
    DIMENSIONS,
    DIMENSION_WEIGHTS,
    TOP_K_EVIDENCE_PER_DIMENSION,
    MIN_EVIDENCE_PER_DIMENSION,
    HISTORY_LOOKBACK_YEARS,
    PURPOSE_DRIVEN_THRESHOLD_0_100,
    EXCLUDE_HUMAN_REVIEW_QUEUE,
    SOURCE_WEIGHTS,
    DEFAULT_SOURCE_WEIGHT,
)

from company_purpose_score_v1.rubric_config import DIMENSION_QUERIES
from company_purpose_score_v1.prompt_templates import (
    SYSTEM_PROMPT,
    build_company_purpose_prompt,
)


def load_env_if_available() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Run: pip install openai python-dotenv scikit-learn") from exc

    api_key = os.getenv(LLM_API_KEY_ENV_VAR)
    if not api_key:
        raise EnvironmentError(f"Missing API key: {LLM_API_KEY_ENV_VAR}")

    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    model = os.getenv(LLM_MODEL_ENV_VAR)
    if not model:
        raise EnvironmentError(
            f"Missing model env var: {LLM_MODEL_ENV_VAR}\n"
            "Example: OPENAI_MODEL=gpt-4o-mini"
        )
    return model


def get_text(row: pd.Series) -> str:
    for col in TEXT_COL_CANDIDATES:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return ""


def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def minmax(series: pd.Series) -> pd.Series:
    series = to_numeric(series)
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def validate_input(df: pd.DataFrame) -> None:
    required = [ID_COL, COMPANY_COL, BUCKET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    text_cols = [col for col in TEXT_COL_CANDIDATES if col in df.columns]
    if not text_cols:
        raise ValueError(f"No text column found. Expected one of {TEXT_COL_CANDIDATES}")


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df[COMPANY_COL].notna()].copy()
    df[COMPANY_COL] = df[COMPANY_COL].astype(str)

    if YEAR_COL in df.columns:
        df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    else:
        df[YEAR_COL] = pd.NA

    if EXCLUDE_HUMAN_REVIEW_QUEUE and "final_needs_human_review" in df.columns:
        df = df[~df["final_needs_human_review"].fillna(False)].copy()

    df["text_for_prompt"] = df.apply(get_text, axis=1)

    df = df[df["text_for_prompt"].str.strip() != ""].copy()
    df = df[df[BUCKET_COL].isin(DIMENSIONS)].copy()

    return df


def build_targets(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    valid_year_df = df[df[YEAR_COL].notna()].copy()

    targets = (
        valid_year_df[[COMPANY_COL, YEAR_COL]]
        .drop_duplicates()
        .sort_values([COMPANY_COL, YEAR_COL])
        .reset_index(drop=True)
    )

    targets[YEAR_COL] = targets[YEAR_COL].astype(int)

    if limit is not None:
        targets = targets.head(limit).copy()

    return targets


def filter_candidates_for_target(
    df: pd.DataFrame,
    company: str,
    year: int,
    dimension: str,
) -> pd.DataFrame:
    candidates = df[
        (df[COMPANY_COL] == company)
        & (df[BUCKET_COL] == dimension)
    ].copy()

    if candidates.empty:
        return candidates

    if dimension == "history_consistency":
        year_mask = (
            candidates[YEAR_COL].isna()
            | (
                (candidates[YEAR_COL] <= year)
                & (candidates[YEAR_COL] >= year - HISTORY_LOOKBACK_YEARS)
            )
        )
    else:
        year_mask = (
            candidates[YEAR_COL].isna()
            | (candidates[YEAR_COL] == year)
        )

    return candidates[year_mask].copy()


def compute_tfidf_similarity(texts: list[str], query: str) -> list[float]:
    if not texts:
        return []

    corpus = texts + [query]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(X[:-1], X[-1]).flatten()
        return sims.tolist()
    except ValueError:
        return [0.0] * len(texts)


def source_weight(source: Any) -> float:
    key = str(source).strip().lower()
    return SOURCE_WEIGHTS.get(key, DEFAULT_SOURCE_WEIGHT)


def rank_dimension_evidence(
    candidates: pd.DataFrame,
    dimension: str,
    top_k: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    candidates = candidates.copy()

    texts = candidates["text_for_prompt"].fillna("").astype(str).tolist()
    query = DIMENSION_QUERIES[dimension]
    candidates["rag_tfidf_similarity"] = compute_tfidf_similarity(texts, query)

    if "bucket_score" in candidates.columns:
        candidates["bucket_score_norm"] = minmax(candidates["bucket_score"])
    else:
        candidates["bucket_score_norm"] = 0.5

    candidates["llm_credibility_norm"] = (
        to_numeric(candidates.get("llm_credibility_score", pd.Series(0, index=candidates.index))) / 5
    )

    candidates["llm_relevance_norm"] = (
        to_numeric(candidates.get("llm_bucket_relevance_score", pd.Series(0, index=candidates.index))) / 5
    )

    if "top_score_margin" in candidates.columns:
        candidates["margin_norm"] = minmax(candidates["top_score_margin"])
    else:
        candidates["margin_norm"] = 0.5

    candidates["source_weight"] = candidates.get("source", "").apply(source_weight)

    if "final_needs_human_review" in candidates.columns:
        candidates["human_review_penalty"] = candidates["final_needs_human_review"].fillna(False).astype(int) * 0.15
    else:
        candidates["human_review_penalty"] = 0.0

    if dimension == "purpose_articulation" and "llm_pa_tone_context_score" in candidates.columns:
        candidates["pa_tone_norm"] = to_numeric(candidates["llm_pa_tone_context_score"]) / 5
    else:
        candidates["pa_tone_norm"] = 0.0

    if dimension == "purpose_articulation":
        candidates["rag_weight"] = (
            0.25 * candidates["rag_tfidf_similarity"]
            + 0.20 * candidates["bucket_score_norm"]
            + 0.20 * candidates["llm_credibility_norm"]
            + 0.15 * candidates["llm_relevance_norm"]
            + 0.10 * candidates["pa_tone_norm"]
            + 0.05 * candidates["source_weight"]
            + 0.05 * candidates["margin_norm"]
            - candidates["human_review_penalty"]
        )
    else:
        candidates["rag_weight"] = (
            0.30 * candidates["rag_tfidf_similarity"]
            + 0.25 * candidates["bucket_score_norm"]
            + 0.20 * candidates["llm_credibility_norm"]
            + 0.15 * candidates["llm_relevance_norm"]
            + 0.05 * candidates["source_weight"]
            + 0.05 * candidates["margin_norm"]
            - candidates["human_review_penalty"]
        )

    candidates["rag_weight"] = candidates["rag_weight"].clip(lower=0)

    ranked = candidates.sort_values(
        by=["rag_weight", "llm_credibility_norm", "bucket_score_norm"],
        ascending=[False, False, False],
    ).head(top_k)

    return ranked


def build_evidence_pack_for_target(
    df: pd.DataFrame,
    company: str,
    year: int,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    pack = {}
    pack_rows = []

    for dimension in DIMENSIONS:
        candidates = filter_candidates_for_target(
            df=df,
            company=company,
            year=year,
            dimension=dimension,
        )

        ranked = rank_dimension_evidence(
            candidates=candidates,
            dimension=dimension,
            top_k=TOP_K_EVIDENCE_PER_DIMENSION,
        )

        rows = ranked.to_dict(orient="records")
        pack[dimension] = rows

        for rank, row in enumerate(rows, start=1):
            row_copy = dict(row)
            row_copy["target_company"] = company
            row_copy["target_year"] = year
            row_copy["target_dimension"] = dimension
            row_copy["dimension_rank"] = rank
            pack_rows.append(row_copy)

    pack_df = pd.DataFrame(pack_rows)

    return pack, pack_df


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty model response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start : end + 1])


def clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def normalize_score_result(raw: dict[str, Any], company: str, year: int) -> dict[str, Any]:
    pa = clamp(raw.get("pa_final_score"), 0, 5, 0)
    hc = clamp(raw.get("hc_final_score"), 0, 5, 0)
    sa = clamp(raw.get("sa_final_score"), 0, 5, 0)

    calculated_0_100 = (
        DIMENSION_WEIGHTS["purpose_articulation"] * pa
        + DIMENSION_WEIGHTS["history_consistency"] * hc
        + DIMENSION_WEIGHTS["strategy_alignment"] * sa
    ) / 5 * 100

    model_score = clamp(
        raw.get("company_purpose_score_0_100"),
        0,
        100,
        calculated_0_100,
    )

    return {
        "company": company,
        "year": year,
        "pa_final_score": round(pa, 2),
        "hc_final_score": round(hc, 2),
        "sa_final_score": round(sa, 2),
        "company_purpose_score_0_100": round(model_score, 2),
        "calculated_purpose_score_0_100": round(calculated_0_100, 2),
        "purpose_driven_label": bool(
            model_score >= PURPOSE_DRIVEN_THRESHOLD_0_100
            and min(pa, hc, sa) >= 2
        ),
        "confidence": clamp(raw.get("confidence"), 0, 1, 0),
        "pa_reason": str(raw.get("pa_reason", ""))[:800],
        "hc_reason": str(raw.get("hc_reason", ""))[:800],
        "sa_reason": str(raw.get("sa_reason", ""))[:800],
        "overall_reason": str(raw.get("overall_reason", ""))[:1000],
        "key_supporting_chunk_ids": json.dumps(raw.get("key_supporting_chunk_ids", []), ensure_ascii=False),
        "weak_or_contradictory_chunk_ids": json.dumps(raw.get("weak_or_contradictory_chunk_ids", []), ensure_ascii=False),
        "needs_human_review": bool(raw.get("needs_human_review", False)),
    }


def call_openai_company_score(
    client,
    model: str,
    prompt: str,
    company: str,
    year: int,
    max_retries: int = 3,
) -> tuple[dict[str, Any], str, str]:
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
            normalized = normalize_score_result(parsed, company=company, year=year)

            return normalized, raw_text, ""

        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(2 * attempt)

    fallback = normalize_score_result({}, company=company, year=year)
    fallback["needs_human_review"] = True

    return fallback, "", last_error


def count_evidence_by_dimension(pack: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {
        f"{dimension}_evidence_count": len(rows)
        for dimension, rows in pack.items()
    }


def process_targets(
    df: pd.DataFrame,
    targets: pd.DataFrame,
    client,
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    evidence_pack_rows = []

    total = len(targets)

    for idx, target in targets.iterrows():
        company = str(target[COMPANY_COL])
        year = int(target[YEAR_COL])

        pack, pack_df = build_evidence_pack_for_target(df, company=company, year=year)

        evidence_counts = count_evidence_by_dimension(pack)

        prompt = build_company_purpose_prompt(
            company=company,
            year=year,
            evidence_pack=pack,
        )

        score_result, raw_response, error_message = call_openai_company_score(
            client=client,
            model=model,
            prompt=prompt,
            company=company,
            year=year,
        )

        score_result.update(evidence_counts)
        score_result["llm_provider"] = LLM_PROVIDER
        score_result["llm_model"] = model
        score_result["purpose_score_run_version"] = RUN_VERSION
        score_result["purpose_score_timestamp"] = datetime.now().isoformat(timespec="seconds")
        score_result["score_status"] = "ok" if not error_message else "error"
        score_result["score_error_message"] = error_message
        score_result["raw_llm_response"] = raw_response

        # Evidence insufficiency flag
        for dimension in DIMENSIONS:
            count_col = f"{dimension}_evidence_count"
            if score_result[count_col] < MIN_EVIDENCE_PER_DIMENSION:
                score_result["needs_human_review"] = True

        score_rows.append(score_result)

        if not pack_df.empty:
            evidence_pack_rows.append(pack_df)

        print(f"[PROGRESS] {idx + 1:,}/{total:,}: {company} {year}")

    scores_df = pd.DataFrame(score_rows)
    evidence_df = (
        pd.concat(evidence_pack_rows, ignore_index=True)
        if evidence_pack_rows
        else pd.DataFrame()
    )

    return scores_df, evidence_df


def aggregate_company_scores(company_year_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for company, group in company_year_df.groupby("company"):
        row = {
            "company": company,
            "years_scored": int(group["year"].nunique()),
            "company_purpose_score_0_100": round(group["company_purpose_score_0_100"].mean(), 2),
            "pa_final_score": round(group["pa_final_score"].mean(), 2),
            "hc_final_score": round(group["hc_final_score"].mean(), 2),
            "sa_final_score": round(group["sa_final_score"].mean(), 2),
            "confidence": round(group["confidence"].mean(), 3),
            "purpose_driven_label": bool(
                group["company_purpose_score_0_100"].mean() >= PURPOSE_DRIVEN_THRESHOLD_0_100
            ),
            "needs_human_review": bool(group["needs_human_review"].any()),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by="company_purpose_score_0_100",
        ascending=False,
    )


def write_diagnostics(company_year_df: pd.DataFrame, evidence_pack_df: pd.DataFrame) -> None:
    lines = []

    lines.append("COMPANY PURPOSE SCORE V1 DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"run_version: {RUN_VERSION}")
    lines.append(f"rows scored: {len(company_year_df):,}")
    lines.append(f"evidence rows used: {len(evidence_pack_df):,}")
    lines.append("")

    lines.append("SCORE SUMMARY")
    lines.append("=" * 80)
    for col in [
        "pa_final_score",
        "hc_final_score",
        "sa_final_score",
        "company_purpose_score_0_100",
        "confidence",
    ]:
        if col in company_year_df.columns:
            s = pd.to_numeric(company_year_df[col], errors="coerce")
            lines.append(
                f"{col}: mean={s.mean():.4f}, median={s.median():.4f}, "
                f"min={s.min():.4f}, max={s.max():.4f}"
            )

    lines.append("")

    lines.append("PURPOSE LABEL DISTRIBUTION")
    lines.append("=" * 80)
    if "purpose_driven_label" in company_year_df.columns:
        for label, count in company_year_df["purpose_driven_label"].value_counts().items():
            lines.append(f"{label}: {count:,}")

    lines.append("")

    lines.append("HUMAN REVIEW")
    lines.append("=" * 80)
    if "needs_human_review" in company_year_df.columns:
        count = int(company_year_df["needs_human_review"].sum())
        rate = count / len(company_year_df) if len(company_year_df) else 0
        lines.append(f"needs_human_review: {count:,}")
        lines.append(f"review_rate: {rate:.4f}")

    path = PHASE4_OUTPUT_DIR / OUTPUT_FILES["diagnostics"]
    path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[EXPORT] diagnostics: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 company-year purpose scoring."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of company-year targets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_env_if_available()

    if not PHASE3_INPUT_FILE.exists():
        raise FileNotFoundError(f"Phase 3 input not found: {PHASE3_INPUT_FILE}")

    PHASE4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    company_year_path = PHASE4_OUTPUT_DIR / OUTPUT_FILES["company_year_scores"]
    company_path = PHASE4_OUTPUT_DIR / OUTPUT_FILES["company_scores"]
    evidence_pack_path = PHASE4_OUTPUT_DIR / OUTPUT_FILES["evidence_pack"]

    if company_year_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {company_year_path}\nUse --force to overwrite."
        )

    print(f"[LOAD] {PHASE3_INPUT_FILE}")
    df = pd.read_csv(PHASE3_INPUT_FILE)

    validate_input(df)
    df = clean_input(df)

    targets = build_targets(df, limit=args.limit)

    print(f"[INFO] evidence rows: {len(df):,}")
    print(f"[INFO] company-year targets: {len(targets):,}")

    client = get_openai_client()
    model = get_model_name()

    company_year_df, evidence_pack_df = process_targets(
        df=df,
        targets=targets,
        client=client,
        model=model,
    )

    company_df = aggregate_company_scores(company_year_df)

    company_year_df.to_csv(company_year_path, index=False)
    company_df.to_csv(company_path, index=False)
    evidence_pack_df.to_csv(evidence_pack_path, index=False)

    print(f"[EXPORT] company-year scores: {company_year_path}")
    print(f"[EXPORT] company scores     : {company_path}")
    print(f"[EXPORT] evidence pack      : {evidence_pack_path}")

    write_diagnostics(company_year_df, evidence_pack_df)

    print("\nDONE: Phase 4 company purpose scoring complete.")


if __name__ == "__main__":
    main()