"""Optional LLM and embedding-based judging layer."""

from __future__ import annotations

import math

from config.scale_definitions import DIMENSION_DEFINITIONS, EMBEDDING_PROTOTYPES
from llm.openai_client import OpenAIClient


def run_optional_openai_judging(company_name, page_features, heuristic_scores, api_key=None):
    """Run optional embedding support scoring and LLM rubric review."""
    client = OpenAIClient(api_key=api_key)
    if not client.enabled:
        return {"enabled": False, "status": "skipped_no_api_key"}

    candidate_snippets = _collect_candidate_snippets(page_features)
    if not candidate_snippets:
        return {"enabled": True, "status": "skipped_no_candidate_snippets"}

    embedding_result = _embedding_support(client, candidate_snippets)
    llm_result = _llm_rubric_review(client, company_name, heuristic_scores, embedding_result)
    return {
        "enabled": True,
        "status": "completed",
        "embedding_support": embedding_result,
        "llm_review": llm_result,
    }


def _collect_candidate_snippets(page_features):
    snippets = []
    for page in sorted(page_features, key=lambda item: item["page_signal_score"], reverse=True)[:12]:
        snippet_texts = page.get("purpose_sentences", [])[:3]
        if not snippet_texts:
            snippet_texts = [page.get("title", "")]
        for snippet in snippet_texts:
            if snippet:
                snippets.append(
                    {
                        "title": page.get("title", ""),
                        "url": page.get("url", ""),
                        "source_bucket": page.get("source_bucket", "other"),
                        "text": snippet[:1200],
                    }
                )
    deduped = []
    seen = set()
    for snippet in snippets:
        key = (snippet["title"], snippet["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snippet)
    return deduped[:20]


def _embedding_support(client, candidate_snippets):
    prototype_inputs = []
    dimension_lookup = []
    for dimension, prototypes in EMBEDDING_PROTOTYPES.items():
        for prototype in prototypes:
            prototype_inputs.append(prototype)
            dimension_lookup.append(dimension)
    snippet_inputs = [snippet["text"] for snippet in candidate_snippets]
    response = client.create_embeddings(
        prototype_inputs + snippet_inputs,
        model="text-embedding-3-large",
    )
    vectors = [item["embedding"] for item in response["data"]]
    prototype_vectors = vectors[: len(prototype_inputs)]
    snippet_vectors = vectors[len(prototype_inputs) :]

    dimension_scores = {}
    dimension_best_snippet = {}
    for prototype_index, dimension in enumerate(dimension_lookup):
        best_score = 0.0
        best_snippet_index = None
        for snippet_index, snippet_vector in enumerate(snippet_vectors):
            score = _cosine_similarity(prototype_vectors[prototype_index], snippet_vector)
            if score > best_score:
                best_score = score
                best_snippet_index = snippet_index
        previous = dimension_scores.get(dimension, 0.0)
        if best_score > previous:
            dimension_scores[dimension] = round(best_score, 4)
            if best_snippet_index is not None:
                dimension_best_snippet[dimension] = candidate_snippets[best_snippet_index]
    return {
        "model": "text-embedding-3-large",
        "dimension_scores": dimension_scores,
        "best_snippets": dimension_best_snippet,
    }


def _llm_rubric_review(client, company_name, heuristic_scores, embedding_result):
    schema = {
        "name": "purpose_driven_review",
        "schema": {
            "type": "object",
            "properties": {
                "overall_adjustment": {
                    "type": "object",
                    "properties": {
                        "delta": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["delta", "reason"],
                    "additionalProperties": False,
                },
                "dimensions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "group": {"type": "string"},
                            "dimension": {"type": "string"},
                            "score": {"type": "integer"},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                            "evidence_title": {"type": "string"},
                        },
                        "required": ["group", "dimension", "score", "confidence", "reason", "evidence_title"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["overall_adjustment", "dimensions"],
            "additionalProperties": False,
        },
    }
    instructions = (
        "You are a strict corporate purpose evaluator. Use only the supplied rubric, "
        "heuristic scores, embedding support, and evidence titles. Be conservative: "
        "do not give 5 unless the evidence clearly supports a top-tier score. "
        "Return JSON only."
    )
    input_text = _build_llm_input(company_name, heuristic_scores, embedding_result)
    return client.create_structured_response(
        model="gpt-5.4-mini",
        instructions=instructions,
        input_text=input_text,
        schema=schema,
        reasoning_effort="low",
    )


def _build_llm_input(company_name, heuristic_scores, embedding_result):
    lines = [f"Company: {company_name}", "", "Rubric:"]
    for group_name, group_meta in DIMENSION_DEFINITIONS.items():
        lines.append(f"{group_name}: {group_meta['title']}")
        for dimension_name, dimension_meta in group_meta["dimensions"].items():
            lines.append(f"- {dimension_name}: {dimension_meta['question']}")
            for score, description in dimension_meta["score_descriptions"].items():
                lines.append(f"  score {score}: {description}")
    lines.append("")
    lines.append("Heuristic scores:")
    lines.append(str(heuristic_scores))
    lines.append("")
    lines.append("Embedding support:")
    lines.append(str(embedding_result))
    return "\n".join(lines)


def _cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)
