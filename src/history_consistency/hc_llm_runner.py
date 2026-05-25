# src/history_consistency/hc_llm_runner.py

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from history_consistency.hc_config import (
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE,
    HC_SCORE_MIN,
    HC_SCORE_MAX,
)
from history_consistency.hc_prompt_builder import (
    build_hc_prompt,
    build_empty_hc_response,
)
from history_consistency.hc_evidence_score import clamp


@dataclass(frozen=True)
class HCLLMResult:
    """
    Company-level HC LLM review result.
    """

    company: str
    hc_score_0_5: float
    rationale: str
    evidence_used: List[str]
    confidence: str
    needs_human_review: bool
    raw_response: str
    parse_success: bool
    error_message: str = ""


def extract_json_object(text: str) -> Dict[str, object]:
    """
    Extract first JSON object from LLM text.

    Handles cases where model wraps JSON in markdown fences.
    """
    if text is None:
        raise ValueError("Empty LLM response.")

    raw = str(text).strip()

    # Remove common code fences.
    raw = raw.replace("```json", "").replace("```", "").strip()

    # First try direct parse.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Then try extracting {...}.
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")

    candidate = match.group(0)
    parsed = json.loads(candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")

    return parsed


def normalize_confidence(value: object) -> str:
    """
    Normalize confidence label.
    """
    text = str(value or "").strip().lower()

    if text in {"low", "medium", "high"}:
        return text

    return "medium"


def normalize_evidence_used(value: object) -> List[str]:
    """
    Normalize evidence_used field into a list of strings.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]

    if isinstance(value, str):
        if not value.strip():
            return []
        return [value.strip()]

    return []


def validate_and_normalize_llm_json(
    company: str,
    parsed: Dict[str, object],
    raw_response: str,
) -> HCLLMResult:
    """
    Validate parsed JSON and convert to HCLLMResult.
    """
    score = clamp(
        parsed.get("hc_score_0_5", 0),
        HC_SCORE_MIN,
        HC_SCORE_MAX,
    )

    rationale = str(parsed.get("rationale", "") or "").strip()
    evidence_used = normalize_evidence_used(parsed.get("evidence_used"))
    confidence = normalize_confidence(parsed.get("confidence"))
    needs_human_review = bool(parsed.get("needs_human_review", False))

    if not rationale:
        rationale = "No rationale provided by model."
        needs_human_review = True

    return HCLLMResult(
        company=company,
        hc_score_0_5=score,
        rationale=rationale,
        evidence_used=evidence_used,
        confidence=confidence,
        needs_human_review=needs_human_review,
        raw_response=raw_response,
        parse_success=True,
        error_message="",
    )


def error_llm_result(
    company: str,
    error_message: str,
    raw_response: str = "",
) -> HCLLMResult:
    """
    Build failed LLM result.
    """
    return HCLLMResult(
        company=company,
        hc_score_0_5=0.0,
        rationale="LLM review failed or response could not be parsed.",
        evidence_used=[],
        confidence="low",
        needs_human_review=True,
        raw_response=raw_response,
        parse_success=False,
        error_message=error_message,
    )


def call_gemini(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    api_key_env: str = "GEMINI_API_KEY",
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Call Gemini model using google.generativeai.

    Requirements:
    - pip install google-generativeai
    - set GEMINI_API_KEY in environment or .env loading logic outside this module
    """
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise EnvironmentError(
            f"Missing API key environment variable: {api_key_env}"
        )

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is not installed. "
            "Install with: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )

    response = model.generate_content(prompt)

    if not hasattr(response, "text") or response.text is None:
        raise RuntimeError("Gemini response did not contain text.")

    return str(response.text)


def _load_anthropic_key_from_local_config() -> str:
    try:
        from configs.config import ANTHROPIC_API_KEY as local_key
    except Exception:
        return ""

    return str(local_key or "").strip()


def _load_openai_key_from_local_config() -> str:
    try:
        from configs.config import OPENAI_API_KEY as local_key
    except Exception:
        return ""

    return str(local_key or "").strip()


def call_openai(
    prompt: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = 1000,
) -> str:
    """
    Call OpenAI Chat Completions API and request a JSON object response.
    """
    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or _load_openai_key_from_local_config()
    )

    if not api_key:
        raise EnvironmentError("Missing API key environment variable: OPENAI_API_KEY")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is not installed. Install with: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful corporate history consistency scoring analyst. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content or ""


def call_claude(
    prompt: str,
    model_name: str = "claude-opus-4-1-20250805",
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = 1000,
) -> str:
    """
    Call Claude through the Anthropic Messages API.
    """
    api_key = (
        os.getenv("ANTHROPIC_API_KEY", "").strip()
        or _load_anthropic_key_from_local_config()
    )

    if not api_key:
        raise EnvironmentError("Missing API key environment variable: ANTHROPIC_API_KEY")

    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": (
            "You are a careful corporate history consistency scoring analyst. "
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


def run_hc_llm_for_company(
    company: str,
    evidence_df: pd.DataFrame,
    provider: str = "gemini",
    model_name: str = "gemini-2.5-flash",
    max_retries: int = LLM_MAX_RETRIES,
    sleep_seconds: float = 1.0,
    max_chars_per_evidence: int = 1400,
) -> HCLLMResult:
    """
    Run company-level HC LLM scoring.

    Provider currently supported:
    - openai
    - gemini
    - claude

    If evidence_df is empty, return deterministic empty result.
    """
    if evidence_df.empty:
        empty = build_empty_hc_response(company)
        return HCLLMResult(
            company=company,
            hc_score_0_5=0.0,
            rationale=str(empty["rationale"]),
            evidence_used=[],
            confidence="low",
            needs_human_review=True,
            raw_response=json.dumps(empty),
            parse_success=True,
            error_message="",
        )

    prompt = build_hc_prompt(
        company=company,
        evidence_df=evidence_df,
        max_chars_per_evidence=max_chars_per_evidence,
    )

    last_error = ""

    for attempt in range(1, max_retries + 1):
        raw_response = ""

        try:
            if provider == "openai":
                raw_response = call_openai(
                    prompt=prompt,
                    model_name=model_name,
                )
            elif provider == "gemini":
                raw_response = call_gemini(
                    prompt=prompt,
                    model_name=model_name,
                )
            elif provider == "claude":
                raw_response = call_claude(
                    prompt=prompt,
                    model_name=model_name,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            parsed = extract_json_object(raw_response)
            return validate_and_normalize_llm_json(
                company=company,
                parsed=parsed,
                raw_response=raw_response,
            )

        except Exception as exc:
            last_error = f"Attempt {attempt}/{max_retries} failed: {exc}"

            if attempt < max_retries:
                time.sleep(sleep_seconds)

    return error_llm_result(
        company=company,
        error_message=last_error,
    )


def llm_result_to_dict(result: HCLLMResult) -> Dict[str, object]:
    """
    Convert HCLLMResult to dict.
    """
    return {
        "company": result.company,
        "hc_llm_score_0_5": result.hc_score_0_5,
        "hc_llm_rationale": result.rationale,
        "hc_llm_evidence_used": json.dumps(result.evidence_used),
        "hc_llm_confidence": result.confidence,
        "hc_llm_needs_human_review": result.needs_human_review,
        "hc_llm_parse_success": result.parse_success,
        "hc_llm_error_message": result.error_message,
        "hc_llm_raw_response": result.raw_response,
    }


def run_hc_llm_for_all_companies(
    evidence_df: pd.DataFrame,
    provider: str = "gemini",
    model_name: str = "gemini-2.5-flash",
    max_retries: int = LLM_MAX_RETRIES,
    max_companies: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run HC LLM scoring for all companies in selected evidence dataframe.

    Output:
    one row per company with company-level LLM score and rationale.
    """
    if evidence_df.empty:
        return pd.DataFrame()

    if "company" not in evidence_df.columns:
        raise ValueError("Evidence dataframe must contain 'company' column.")

    rows: List[Dict[str, object]] = []

    grouped = list(evidence_df.groupby("company", dropna=False))

    if max_companies is not None:
        grouped = grouped[:max_companies]

    for company, group in grouped:
        company_name = str(company).strip()

        result = run_hc_llm_for_company(
            company=company_name,
            evidence_df=group.copy(),
            provider=provider,
            model_name=model_name,
            max_retries=max_retries,
        )

        rows.append(llm_result_to_dict(result))

    return pd.DataFrame(rows)


def attach_company_llm_scores_to_evidence(
    evidence_df: pd.DataFrame,
    llm_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach company-level HC LLM result to each evidence row.

    HC has one question, so the company-level rubric score is propagated to
    each selected evidence row before evidence quality / redundancy adjustment.
    """
    if evidence_df.empty:
        return evidence_df.copy()

    if llm_results_df.empty:
        out = evidence_df.copy()
        out["hc_llm_score_0_5"] = pd.NA
        out["hc_llm_rationale"] = ""
        out["hc_llm_confidence"] = ""
        out["hc_llm_needs_human_review"] = True
        out["hc_llm_parse_success"] = False
        return out

    keep_cols = [
        "company",
        "hc_llm_score_0_5",
        "hc_llm_rationale",
        "hc_llm_confidence",
        "hc_llm_needs_human_review",
        "hc_llm_parse_success",
        "hc_llm_error_message",
    ]
    keep_cols = [c for c in keep_cols if c in llm_results_df.columns]

    out = evidence_df.merge(
        llm_results_df[keep_cols],
        on="company",
        how="left",
    )

    return out


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "company": ["Sample Company", "Sample Company"],
            "chunk_id": ["c1", "c2"],
            "year": [2018, 2024],
            "source": ["edgar", "edgar"],
            "section": ["letter to shareholders", "business"],
            "text_clean": [
                "Since our founding, our purpose has been to improve access to technology.",
                "We remain committed to improving access to technology globally.",
            ],
            "hc_rank_score": [0.91, 0.88],
            "hc_base_evidence_score_0_1": [0.83, 0.79],
        }
    )

    prompt = build_hc_prompt("Sample Company", sample)
    print(prompt[:2000])
