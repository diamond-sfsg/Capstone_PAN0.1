"""CSV ingestion helpers."""

from __future__ import annotations

import csv
from pathlib import Path


def load_csv(path):
    """Return normalized CSV records from the given path."""
    source = Path(path)
    records = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            records.append(
                {
                    "source_file": str(source),
                    "source_type": "csv",
                    "record_id": f"{source.stem}-{index}",
                    "url": row.get("url", ""),
                    "path": row.get("path", ""),
                    "title": row.get("title", ""),
                    "meta_description": row.get("meta_description", ""),
                    "headings": row.get("headings", ""),
                    "text": row.get("text", ""),
                    "status_code": row.get("status_code", ""),
                    "is_probably_relevant": str(row.get("is_probably_relevant", "")).lower() == "true",
                    "url_importance_score": _to_number(row.get("url_importance_score")),
                    "purpose_signal_count": _to_number(row.get("purpose_signal_count")),
                    "purpose_signal_hits": row.get("purpose_signal_hits", ""),
                    "text_length": _to_number(row.get("text_length")),
                }
            )
    return records


def _to_number(value):
    try:
        if value is None or value == "":
            return 0
        return float(value)
    except (TypeError, ValueError):
        return 0

