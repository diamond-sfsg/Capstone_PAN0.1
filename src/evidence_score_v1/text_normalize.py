from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    if text is None:
        text = ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_retrieval_text_columns(df):
    out = df.copy()
    text = out["text_clean"] if "text_clean" in out.columns else ""
    out["text_for_match"] = text.fillna("").map(normalize_for_matching)

    metadata_parts = []
    for col in ("company", "source", "section", "subsection"):
        if col in out.columns:
            metadata_parts.append(out[col].fillna("").astype(str))

    if metadata_parts:
        metadata_text = metadata_parts[0]
        for part in metadata_parts[1:]:
            metadata_text = metadata_text.str.cat(part, sep=" ")
        out["retrieval_text"] = (
            metadata_text.str.cat(out["text_clean"].fillna("").astype(str), sep=" ")
            .map(normalize_for_matching)
        )
    else:
        out["retrieval_text"] = out["text_for_match"]

    return out
