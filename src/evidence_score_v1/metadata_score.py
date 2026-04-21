from __future__ import annotations

import pandas as pd


def compute_metadata_score(
    section: str,
    subsection: str,
    source: str,
    preferred_sections: list[str],
    preferred_sources: list[str],
) -> float:
    section_text = f"{section or ''} {subsection or ''}".lower()
    source_text = (source or "").lower()

    score = 0.0

    section_hits = sum(1 for s in preferred_sections if s.lower() in section_text)
    source_hits = sum(1 for s in preferred_sources if s.lower() == source_text)

    if section_hits > 0:
        score += min(0.10, 0.03 * section_hits)

    if source_hits > 0:
        score += 0.05

    return min(score, 0.15)


def add_metadata_scores(
    df: pd.DataFrame,
    preferred_sections: list[str],
    preferred_sources: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out["metadata_score"] = out.apply(
        lambda row: compute_metadata_score(
            section=row.get("section", ""),
            subsection=row.get("subsection", ""),
            source=row.get("source", ""),
            preferred_sections=preferred_sections,
            preferred_sources=preferred_sources,
        ),
        axis=1,
    )
    return out