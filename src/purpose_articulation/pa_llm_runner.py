from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

from purpose_articulation.pa_config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_FALLBACK_TO_MOCK,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PA_TONE_BONUS_MAX,
    PA_TONE_BONUS_MIN,
)
from purpose_articulation.pa_evidence_score import (
    generic_branding_penalty,
    rule_based_pa_tone_bonus,
)
from purpose_articulation.pa_prompt_builder import (
    build_pa_evidence_prompt,
    build_pa_evidence_set_prompt,
)


def clamp(value: float | int | None, lower: float, upper: float) -> float:
    if value is None:
        return lower

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return lower

    if pd.isna(value_float):
        return lower

    return max(lower, min(upper, value_float))


def extract_json(text: str | None) -> dict[str, Any]:
    """
    Extract JSON object from model output.
    """
    if text is None:
        return {}

    text = str(text).strip()

    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def normalize_risk_flags(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, str):
        if not value.strip():
            return []
        return [value.strip()]

    return [str(value)]


def normalize_extracted_purpose(value: Any) -> str:
    """
    Normalize LLM extracted purpose text for evidence retention.
    """
    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(str(item) for item in value if str(item).strip())

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    if text.lower() in {"none", "n/a", "na", "not available", "no purpose found"}:
        return ""

    return text


def extract_purpose_sentence_from_text(text: str) -> str:
    """
    Lightweight mock-mode extraction of a purpose-like sentence.
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""

    sentence_candidates = re.split(r"(?<=[.!?])\s+", text)
    purpose_terms = [
        "our purpose",
        "our mission",
        "we exist to",
        "exists to",
        "we help",
        "we enable",
        "we empower",
        "committed to",
        "improve",
        "serve",
    ]

    for sentence in sentence_candidates:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue
        norm = sentence_clean.lower()
        if any(term in norm for term in purpose_terms):
            return sentence_clean[:320]

    return ""


class PAEvaluator:
    """
    LLM evaluator for PA scoring.

    provider="mock":
        deterministic heuristic scorer for pipeline testing.

    provider="gemini":
        uses google.generativeai and GEMINI_API_KEY.

    provider="openai":
        uses the OpenAI SDK and OPENAI_API_KEY.

    provider="claude":
        uses Anthropic Messages API and ANTHROPIC_API_KEY.
    """

    def __init__(self, provider: str | None = None):
        self.provider = (provider or LLM_PROVIDER or "mock").strip().lower()
        self._gemini_model = None

        if self.provider == "gemini":
            self._setup_gemini()
        elif self.provider == "openai":
            self._setup_openai()
        elif self.provider == "claude":
            self._setup_claude()

    def _setup_gemini(self) -> None:
        if not GEMINI_API_KEY:
            if LLM_FALLBACK_TO_MOCK:
                self.provider = "mock"
                return

            raise RuntimeError("GEMINI_API_KEY is missing.")

        try:
            import google.generativeai as genai
        except ImportError as exc:
            if LLM_FALLBACK_TO_MOCK:
                self.provider = "mock"
                return

            raise ImportError(
                "google-generativeai is not installed. "
                "Install it or use PA_LLM_PROVIDER=mock."
            ) from exc

        genai.configure(api_key=GEMINI_API_KEY)
        self._gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    def _setup_openai(self) -> None:
        api_key = (
            os.getenv("OPENAI_API_KEY", "").strip()
            or OPENAI_API_KEY
            or self._load_openai_key_from_local_config()
        )

        if not api_key:
            if LLM_FALLBACK_TO_MOCK:
                self.provider = "mock"
                return

            raise RuntimeError("OPENAI_API_KEY is missing.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            if LLM_FALLBACK_TO_MOCK:
                self.provider = "mock"
                return

            raise ImportError(
                "openai is not installed. Install it or use PA_LLM_PROVIDER=mock."
            ) from exc

        self._openai_client = OpenAI(api_key=api_key)

    def _load_openai_key_from_local_config(self) -> str:
        try:
            from configs.config import OPENAI_API_KEY as local_key
        except Exception:
            return ""

        return str(local_key or "").strip()

    def _setup_claude(self) -> None:
        api_key = (
            os.getenv("ANTHROPIC_API_KEY", "").strip()
            or ANTHROPIC_API_KEY
            or self._load_anthropic_key_from_local_config()
        )

        if not api_key:
            if LLM_FALLBACK_TO_MOCK:
                self.provider = "mock"
                return

            raise RuntimeError("ANTHROPIC_API_KEY is missing.")

        self._anthropic_api_key = api_key

    def _load_anthropic_key_from_local_config(self) -> str:
        try:
            from configs.config import ANTHROPIC_API_KEY as local_key
        except Exception:
            return ""

        return str(local_key or "").strip()

    def _call_gemini(self, prompt: str) -> str:
        if self._gemini_model is None:
            raise RuntimeError("Gemini model is not initialized.")

        response = self._gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": LLM_TEMPERATURE,
            },
        )

        return getattr(response, "text", "") or ""

    def _call_openai(self, prompt: str) -> str:
        client = getattr(self, "_openai_client", None)
        if client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful corporate purpose scoring analyst. "
                        "Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content or ""

    def _call_claude(self, prompt: str) -> str:
        api_key = getattr(self, "_anthropic_api_key", "")
        if not api_key:
            raise RuntimeError("Claude client is not initialized.")

        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": LLM_MAX_OUTPUT_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "system": (
                "You are a careful corporate purpose scoring analyst. "
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

    def _call_provider_json(self, prompt: str) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None

        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                if self.provider == "gemini":
                    raw_text = self._call_gemini(prompt)
                    parsed = extract_json(raw_text)

                    if parsed:
                        return parsed, raw_text

                    raise ValueError("Could not parse JSON from Gemini response.")

                if self.provider == "openai":
                    raw_text = self._call_openai(prompt)
                    parsed = extract_json(raw_text)

                    if parsed:
                        return parsed, raw_text

                    raise ValueError("Could not parse JSON from OpenAI response.")

                if self.provider == "claude":
                    raw_text = self._call_claude(prompt)
                    parsed = extract_json(raw_text)

                    if parsed:
                        return parsed, raw_text

                    raise ValueError("Could not parse JSON from Claude response.")

                raise ValueError(f"Unsupported LLM provider: {self.provider}")

            except Exception as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))

        if LLM_FALLBACK_TO_MOCK:
            return {}, f"LLM failed; fallback to mock. Error: {last_error}"

        raise RuntimeError(f"LLM call failed: {last_error}")

    def score_evidence(
        self,
        question_id: str,
        evidence_row: dict,
    ) -> tuple[dict[str, Any], str]:
        """
        Score one evidence chunk for PA_Q1 or PA_Q2.
        """
        if self.provider == "mock":
            result = self._mock_score_evidence(question_id, evidence_row)
            raw_output = json.dumps(result, ensure_ascii=False)
            return result, raw_output

        prompt = build_pa_evidence_prompt(question_id, evidence_row)
        parsed, raw_output = self._call_provider_json(prompt)

        if not parsed and LLM_FALLBACK_TO_MOCK:
            result = self._mock_score_evidence(question_id, evidence_row)
            return result, raw_output

        result = {
            "llm_score_0_5": clamp(parsed.get("llm_score_0_5"), 0.0, 5.0),
            "pa_tone_bonus": clamp(
                parsed.get("pa_tone_bonus"),
                PA_TONE_BONUS_MIN,
                PA_TONE_BONUS_MAX,
            ),
            "extracted_purpose": normalize_extracted_purpose(
                parsed.get("extracted_purpose")
            ),
            "support_level": str(parsed.get("support_level", "unknown")),
            "reason": str(parsed.get("reason", "")),
            "risk_flags": normalize_risk_flags(parsed.get("risk_flags")),
        }

        return result, raw_output

    def score_evidence_set(
        self,
        question_id: str,
        evidence_set_df: pd.DataFrame,
        set_quality: dict,
    ) -> tuple[dict[str, Any], str]:
        """
        Score Q3 evidence set.
        """
        if self.provider == "mock":
            result = self._mock_score_evidence_set(evidence_set_df, set_quality)
            raw_output = json.dumps(result, ensure_ascii=False)
            return result, raw_output

        prompt = build_pa_evidence_set_prompt(
            question_id=question_id,
            evidence_set_df=evidence_set_df,
            set_quality=set_quality,
        )

        parsed, raw_output = self._call_provider_json(prompt)

        if not parsed and LLM_FALLBACK_TO_MOCK:
            result = self._mock_score_evidence_set(evidence_set_df, set_quality)
            return result, raw_output

        result = {
            "llm_set_score_0_5": clamp(
                parsed.get("llm_set_score_0_5"),
                0.0,
                5.0,
            ),
            "extracted_purpose": normalize_extracted_purpose(
                parsed.get("extracted_purpose")
            ),
            "support_level": str(parsed.get("support_level", "unknown")),
            "reason": str(parsed.get("reason", "")),
            "risk_flags": normalize_risk_flags(parsed.get("risk_flags")),
        }

        return result, raw_output

    def _mock_score_evidence(
        self,
        question_id: str,
        evidence_row: dict,
    ) -> dict[str, Any]:
        """
        Deterministic fallback scorer for local pipeline testing.

        This is not a replacement for final LLM scoring.
        """
        text = str(evidence_row.get("text_clean", ""))
        norm = text.lower()

        keyword = float(evidence_row.get("keyword_relevance", 0.0) or 0.0)
        context = float(evidence_row.get("context_completeness", 0.0) or 0.0)
        rag = float(evidence_row.get("rag_similarity", 0.0) or 0.0)

        explicit = any(
            phrase in norm
            for phrase in [
                "our purpose",
                "our mission",
                "we exist to",
                "exists to",
                "our vision",
            ]
        )

        stakeholder = any(
            term in norm
            for term in [
                "customers",
                "patients",
                "communities",
                "people",
                "stakeholders",
                "society",
                "employees",
            ]
        )

        impact = any(
            term in norm
            for term in [
                "impact",
                "improve",
                "enable",
                "access",
                "affordable",
                "safe",
                "healthy",
                "sustainable",
                "opportunity",
            ]
        )

        branding_penalty = generic_branding_penalty(text)

        if question_id == "PA_Q1":
            score = (
                0.8
                + 2.0 * keyword
                + 0.8 * int(explicit)
                + 0.6 * int(stakeholder)
                + 0.6 * int(impact)
            )

        elif question_id == "PA_Q2":
            score = (
                0.7
                + 1.4 * context
                + 1.2 * rag
                + 0.7 * int(impact)
                - 1.0 * branding_penalty
            )

        else:
            score = 0.8 + keyword + context

        score = clamp(score, 0.0, 5.0)

        if score >= 4.2:
            support = "very_strong"
        elif score >= 3.3:
            support = "strong"
        elif score >= 2.2:
            support = "moderate"
        elif score >= 1.0:
            support = "weak"
        else:
            support = "none"

        flags = []

        if branding_penalty >= 0.5:
            flags.append("generic_language")

        if context < 0.4:
            flags.append("incomplete_context")

        section = str(evidence_row.get("section", "")).lower()
        if "homepage" in section or "banner" in section:
            flags.append("potential_marketing_context")

        return {
            "llm_score_0_5": score,
            "pa_tone_bonus": rule_based_pa_tone_bonus(text),
            "extracted_purpose": extract_purpose_sentence_from_text(text),
            "support_level": support,
            "reason": (
                "Mock heuristic score for pipeline testing. "
                "Use PA_LLM_PROVIDER=openai or gemini for final rubric scoring."
            ),
            "risk_flags": flags,
        }

    def _mock_score_evidence_set(
        self,
        evidence_set_df: pd.DataFrame,
        set_quality: dict,
    ) -> dict[str, Any]:
        """
        Mock scorer for Q3 evidence-set scoring.
        """
        if evidence_set_df is None or evidence_set_df.empty:
            return {
                "llm_set_score_0_5": 0.0,
                "extracted_purpose": "",
                "support_level": "none",
                "reason": "No evidence set was available.",
                "risk_flags": ["no_evidence"],
            }

        sources = set(
            evidence_set_df.get("normalized_source", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.lower()
            .tolist()
        )

        has_edgar = "edgar" in sources
        has_web = "official_web" in sources

        score = 0.8
        score += 1.2 * float(set_quality.get("source_diversity", 0.0) or 0.0)
        score += 1.2 * float(set_quality.get("formal_document_presence", 0.0) or 0.0)
        score += 1.0 * float(set_quality.get("strategic_section_presence", 0.0) or 0.0)

        if has_edgar and has_web:
            score += 0.6

        score = clamp(score, 0.0, 5.0)

        flags = []

        if not has_edgar:
            flags.append("insufficient_formal_documents")

        if float(set_quality.get("strategic_section_presence", 0.0) or 0.0) == 0.0:
            flags.append("weak_strategy_context")

        if score >= 4.2:
            support = "very_strong"
        elif score >= 3.3:
            support = "strong"
        elif score >= 2.2:
            support = "moderate"
        elif score >= 1.0:
            support = "weak"
        else:
            support = "none"

        return {
            "llm_set_score_0_5": score,
            "extracted_purpose": extract_purpose_sentence_from_text(
                " ".join(
                    evidence_set_df.get("text_clean", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .head(8)
                    .tolist()
                )
            ),
            "support_level": support,
            "reason": (
                "Mock evidence-set score for pipeline testing. "
                "Use PA_LLM_PROVIDER=openai or gemini for final rubric scoring."
            ),
            "risk_flags": flags,
        }
