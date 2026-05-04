from __future__ import annotations

import pandas as pd


def score_metadata(df: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    preferred_sections = [s.lower() for s in cfg.preferred_sections]
    preferred_sources = [s.lower() for s in cfg.preferred_sources]
    for _, row in df.iterrows():
        section = str(row.get("section") or "").lower()
        source = str(row.get("source") or "").lower()
        section_prior = 0.0
        source_prior = 0.0
        if any(marker in section for marker in preferred_sections):
            section_prior = 1.0
        if source in preferred_sources:
            source_prior = 0.5
        rows.append(
            {
                "section_prior_score": section_prior,
                "source_prior_score": source_prior,
                "metadata_total_score": section_prior + source_prior,
            }
        )
    return pd.DataFrame(rows)


def compute_metadata_scores(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = score_metadata(df, cfg)
    out.insert(0, "chunk_id", df["chunk_id"].to_numpy())
    out["metadata_score"] = out["metadata_total_score"]
    return out
