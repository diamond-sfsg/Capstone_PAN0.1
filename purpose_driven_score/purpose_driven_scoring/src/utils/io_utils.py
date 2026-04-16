"""Shared input/output utility functions."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def ensure_directory(path):
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path, payload):
    """Write JSON payload with UTF-8 encoding."""
    target = Path(path)
    ensure_directory(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_csv(path, rows):
    """Write rows to CSV when rows are present."""
    target = Path(path)
    ensure_directory(target.parent)
    rows = list(rows)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target

