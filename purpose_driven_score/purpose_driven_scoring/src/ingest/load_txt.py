"""Text ingestion helpers."""

from __future__ import annotations

from pathlib import Path


def load_txt(path):
    """Return text content from the given path."""
    source = Path(path)
    return [
        {
            "source_file": str(source),
            "source_type": "txt",
            "record_id": source.stem,
            "url": "",
            "path": str(source),
            "title": source.stem,
            "meta_description": "",
            "headings": "",
            "text": source.read_text(encoding="utf-8"),
            "status_code": "",
            "is_probably_relevant": True,
            "url_importance_score": 0,
            "purpose_signal_count": 0,
            "purpose_signal_hits": "",
            "text_length": source.stat().st_size,
        }
    ]

