"""Deduplication utilities."""

from __future__ import annotations

import hashlib


def deduplicate_records(records):
    """Remove duplicate records from a sequence."""
    unique_records = []
    seen = set()
    for record in records:
        fingerprint_parts = [
            str(record.get("url", "")),
            str(record.get("title", "")),
            str(record.get("text", ""))[:2000],
        ]
        fingerprint = hashlib.sha1("||".join(fingerprint_parts).encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_records.append(record)
    return unique_records

