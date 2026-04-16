"""Minimal OpenAI API client using the official Responses and Embeddings endpoints."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAIClient:
    """Small stdlib client for optional LLM and embedding judgments."""

    def __init__(self, api_key=None, base_url="https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    @property
    def enabled(self):
        return bool(self.api_key)

    def create_embeddings(self, inputs, model):
        payload = {
            "model": model,
            "input": inputs,
        }
        return self._post_json("/embeddings", payload)

    def create_structured_response(self, model, instructions, input_text, schema, reasoning_effort="low"):
        payload = {
            "model": model,
            "instructions": instructions,
            "input": input_text,
            "reasoning": {"effort": reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "strict": True,
                    "schema": schema["schema"],
                }
            },
        }
        response = self._post_json("/responses", payload)
        return _extract_response_json(response)

    def _post_json(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc


def _extract_response_json(response):
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return json.loads(content["text"])
            if content.get("type") == "json_schema" and content.get("json"):
                return content["json"]
    text = response.get("output_text")
    if text:
        return json.loads(text)
    raise RuntimeError("Structured response did not contain JSON output.")

