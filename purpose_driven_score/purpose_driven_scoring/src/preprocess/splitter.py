"""Document splitting utilities."""

from __future__ import annotations


def split_text(text, max_words=180):
    """Split raw text into analysis-ready chunks."""
    words = (text or "").split()
    if not words:
        return []
    chunks = []
    for index in range(0, len(words), max_words):
        chunks.append(" ".join(words[index : index + max_words]))
    return chunks

