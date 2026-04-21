from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-]+\b")


def normalize_for_lexical(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_for_lexical(text))


def count_phrase_hits(text: str, phrases: list[str]) -> int:
    text_norm = normalize_for_lexical(text)
    hits = 0
    for phrase in phrases:
        phrase_norm = normalize_for_lexical(phrase)
        if not phrase_norm:
            continue
        hits += text_norm.count(phrase_norm)
    return hits


def compute_lexical_score(
    text: str,
    keywords_core: list[str],
    keywords_support: list[str],
) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0

    token_counts = Counter(tokens)
    text_len = len(tokens)
    text_norm = normalize_for_lexical(text)

    single_core = [k.lower() for k in keywords_core if " " not in k.strip()]
    phrase_core = [k.lower() for k in keywords_core if " " in k.strip()]

    single_support = [k.lower() for k in keywords_support if " " not in k.strip()]
    phrase_support = [k.lower() for k in keywords_support if " " in k.strip()]

    core_hits = sum(token_counts[k] for k in single_core if k in token_counts)
    support_hits = sum(token_counts[k] for k in single_support if k in token_counts)

    core_phrase_hits = count_phrase_hits(text_norm, phrase_core)
    support_phrase_hits = count_phrase_hits(text_norm, phrase_support)

    core_unique = sum(1 for k in single_core if k in token_counts)
    support_unique = sum(1 for k in single_support if k in token_counts)

    core_phrase_unique = sum(1 for k in phrase_core if k in text_norm)
    support_phrase_unique = sum(1 for k in phrase_support if k in text_norm)

    raw_score = (
        2.5 * core_hits +
        1.2 * support_hits +
        3.0 * core_phrase_hits +
        1.5 * support_phrase_hits +
        3.0 * core_unique +
        1.5 * support_unique +
        3.0 * core_phrase_unique +
        1.5 * support_phrase_unique
    )

    return raw_score / math.sqrt(text_len + 20)


def add_lexical_scores(
    df: pd.DataFrame,
    keywords_core: list[str],
    keywords_support: list[str],
    text_col: str = "text_clean",
) -> pd.DataFrame:
    out = df.copy()
    out["lexical_score"] = out[text_col].fillna("").map(
        lambda x: compute_lexical_score(
            text=x,
            keywords_core=keywords_core,
            keywords_support=keywords_support,
        )
    )
    return out