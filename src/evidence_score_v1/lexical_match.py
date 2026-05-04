from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def _phrase_count(text: str, phrases: Iterable[str]) -> tuple[int, int]:
    total_hits = 0
    unique_hits = 0
    for phrase in phrases:
        count = text.count(phrase.lower())
        total_hits += count
        if count > 0:
            unique_hits += 1
    return total_hits, unique_hits


def score_lexical(df: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for text in df["text_for_match"].tolist():
        core_hits, core_unique = _phrase_count(text, cfg.core_phrases)
        support_hits, support_unique = _phrase_count(text, cfg.support_phrases)
        negative_hits, _ = _phrase_count(text, cfg.negative_phrases)
        token_count = max(len(text.split()), 1)
        raw = (
            2.5 * core_hits
            + 3.0 * core_unique
            + 1.25 * support_hits
            + 1.5 * support_unique
            - 2.0 * negative_hits
        ) / math.sqrt(token_count + 20)
        rows.append(
            {
                "core_phrase_hits": core_hits,
                "support_phrase_hits": support_hits,
                "negative_phrase_hits": negative_hits,
                "core_phrase_unique": core_unique,
                "support_phrase_unique": support_unique,
                "lexical_raw_score": float(raw),
            }
        )
    return pd.DataFrame(rows)


def compute_lexical_scores(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = score_lexical(df, cfg)
    out.insert(0, "chunk_id", df["chunk_id"].to_numpy())
    out["lexical_score"] = out["lexical_raw_score"]
    return out
