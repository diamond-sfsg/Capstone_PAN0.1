"""JSON ingestion helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path):
    """Return JSON or JSONL content from the given path."""
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return _load_jsonl(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    else:
        rows = [payload]
    return [_normalize_record(source, index + 1, row) for index, row in enumerate(rows)]


def _load_jsonl(source: Path):
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(_normalize_record(source, index, json.loads(line)))
    return rows


def _normalize_record(source: Path, index: int, row):
    if not isinstance(row, dict):
        row = {"text": str(row)}
    return {
        "source_file": str(source),
        "source_type": source.suffix.lower().lstrip("."),
        "record_id": f"{source.stem}-{index}",
        "url": row.get("url", ""),
        "path": row.get("path", ""),
        "title": row.get("title", ""),
        "meta_description": row.get("meta_description", ""),
        "headings": _normalize_headings(row.get("headings", "")),
        "text": row.get("text", ""),
        "status_code": row.get("status_code", ""),
        "is_probably_relevant": bool(row.get("is_probably_relevant", False)),
        "url_importance_score": _to_number(row.get("url_importance_score")),
        "purpose_signal_count": _to_number(row.get("purpose_signal_count")),
        "purpose_signal_hits": row.get("purpose_signal_hits", ""),
        "text_length": _to_number(row.get("text_length", len(row.get("text", "")))),
    }


def _normalize_headings(headings):
    if isinstance(headings, list):
        return " | ".join(str(item) for item in headings)
    return headings or ""


def _to_number(value):
    try:
        if value is None or value == "":
            return 0
        return float(value)
    except (TypeError, ValueError):
        return 0

