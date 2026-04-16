"""Shared date utility functions."""

from __future__ import annotations

import re


def parse_date(value):
    """Return the incoming value unchanged for now."""
    return value


def extract_years(value: str) -> list[int]:
    """Extract 4-digit years from text."""
    years = []
    for match in re.findall(r"\b(20\d{2}|19\d{2})\b", value or ""):
        year = int(match)
        if year not in years:
            years.append(year)
    return sorted(years)

