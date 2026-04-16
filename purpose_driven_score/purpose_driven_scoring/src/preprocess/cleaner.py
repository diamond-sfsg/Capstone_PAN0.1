"""Text cleaning utilities."""

from __future__ import annotations

import re

from config.scoring_config import BOILERPLATE_PATTERNS
from utils.text_utils import normalize_whitespace


def clean_text(text):
    """Normalize raw text before feature extraction."""
    cleaned = text or ""
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", cleaned)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"(?is)&nbsp;|&#160;", " ", cleaned)
    cleaned = re.sub(r"(?is)accession number:.*?</sec-header>", " ", cleaned)
    cleaned = re.sub(r"(?is)<sec-header>.*?</sec-header>", " ", cleaned)
    cleaned = re.sub(r"(?is)table of contents", " ", cleaned)
    cleaned = re.sub(r"(?is)skip to main.*?skip to footer", " ", cleaned)
    cleaned = re.sub(r"(?is)\bmenu\b\s*\|", " ", cleaned)
    cleaned = re.sub(r"(?is)hello,\s*how can i help\??", " ", cleaned)
    cleaned = re.sub(r"(?is)related content.*$", " ", cleaned)
    cleaned = normalize_whitespace(cleaned)
    lowered = cleaned.lower()
    for pattern in BOILERPLATE_PATTERNS:
        index = lowered.find(pattern)
        if index != -1:
            cleaned = cleaned[:index].strip()
            lowered = cleaned.lower()
    return cleaned
