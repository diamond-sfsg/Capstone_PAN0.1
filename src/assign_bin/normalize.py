# src/assign_bin/normalize.py

from __future__ import annotations

import html
import re
import unicodedata
from typing import List, Tuple

from .config import SHORT_TEXT_THRESHOLD, MAX_CHUNK_TOKENS


WHITESPACE_RE = re.compile(r"\s+")
MULTI_NEWLINE_RE = re.compile(r"\n{2,}")
TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9&/\-'.%]*\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*table of contents\s*$", re.I),
    re.compile(r"^\s*forward[- ]looking statements?\s*$", re.I),
    re.compile(r"^\s*safe harbor\s*$", re.I),
    re.compile(r"^\s*copyright\s+\d{4}", re.I),
    re.compile(r"^\s*all rights reserved\.?\s*$", re.I),
    re.compile(r"^\s*page\s+\d+\s*$", re.I),
    re.compile(r"^\s*read more\s*$", re.I),
    re.compile(r"^\s*learn more\s*$", re.I),
]


def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def remove_control_chars(text: str) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")


def strip_html_artifacts(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def remove_common_artifacts(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "•": " ",
        "▪": " ",
        "●": " ",
        "■": " ",
        "□": " ",
        "\u00a0": " ",
        "\ufeff": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[-_]{4,}", " ", text)
    text = re.sub(r"[=]{4,}", " ", text)
    text = re.sub(r"[|]{2,}", " ", text)

    return text.strip()


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    轻量清洗：
    - 保留年份、数字、百分比、大小写语义
    - 不做 lowercase
    - 不做 stopword removal
    """
    if not text:
        return ""

    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = strip_html_artifacts(text)
    text = remove_common_artifacts(text)
    text = normalize_whitespace(text)
    return text


def normalize_for_match(text: str) -> str:
    """
    给 duplicate / similarity / boilerplate 用
    """
    if not text:
        return ""
    text = clean_text(text)
    text = text.lower()
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_for_similarity(text: str) -> str:
    """
    比 normalize_for_match 更激进：
    - lowercase
    - 去数字
    - 去标点
    """
    if not text:
        return ""
    text = normalize_for_match(text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_RE.findall(text))


def count_chars(text: str) -> int:
    return len(text or "")


def split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p and p.strip()]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = normalize_whitespace(text)
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def is_boilerplate_like(text: str) -> bool:
    if not text:
        return True

    normalized = normalize_for_match(text)

    if len(normalized) < 5:
        return True

    for pattern in BOILERPLATE_PATTERNS:
        if pattern.match(normalized):
            return True

    if normalized in {"home", "menu", "investors", "about us"}:
        return True

    return False


def is_garbled_text(text: str) -> bool:
    """
    轻量乱码检测：
    - replacement char / box chars
    - 过低字母占比
    """
    if not text:
        return True

    bad_chars = {"�", "□", "■", "▪"}
    bad_char_ratio = sum(1 for ch in text if ch in bad_chars) / max(len(text), 1)
    alpha_ratio = sum(1 for ch in text if ch.isalpha()) / max(len(text), 1)

    if bad_char_ratio > 0.01:
        return True

    # 比较长但几乎没有正常字母，通常是脏表格/XBRL/乱码块
    if len(text) > 80 and alpha_ratio < 0.30:
        return True

    return False


def get_quality_flag(text_clean: str) -> str:
    token_count = count_tokens(text_clean)

    if not text_clean.strip():
        return "empty_after_clean"
    if is_garbled_text(text_clean):
        return "garbled_text"
    if is_boilerplate_like(text_clean):
        return "boilerplate_like"
    if token_count < SHORT_TEXT_THRESHOLD:
        return "too_short"
    if token_count > MAX_CHUNK_TOKENS:
        return "too_long"
    return "ok"


def build_chunk_quality_fields(text_clean: str) -> Tuple[int, int, bool, str]:
    token_count = count_tokens(text_clean)
    char_count = count_chars(text_clean)
    is_short_text = token_count < SHORT_TEXT_THRESHOLD
    quality_flag = get_quality_flag(text_clean)
    return token_count, char_count, is_short_text, quality_flag