from __future__ import annotations

import re

import pandas as pd


def score_prompt_patterns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    compiled_groups = {
        group.name: [re.compile(p, flags=re.IGNORECASE) for p in group.patterns]
        for group in cfg.prompt_pattern_groups
    }
    group_weights = {group.name: group.weight for group in cfg.prompt_pattern_groups}

    rows = []
    for text in df["text_for_match"].tolist():
        group_hits = {}
        total_hits = 0
        score = 0.0
        for name, patterns in compiled_groups.items():
            hits = sum(1 for p in patterns if p.search(text))
            group_hits[name] = hits
            total_hits += hits
            if hits:
                score += group_weights[name] * hits
        rows.append(
            {
                "prompt_pattern_hits": total_hits,
                "prompt_pattern_groups_hit": sum(1 for v in group_hits.values() if v > 0),
                "explicit_purpose_pattern_hit": 1 if group_hits.get("explicit_purpose", 0) > 0 else 0,
                "stakeholder_pattern_hit": 1 if group_hits.get("stakeholder_focus", 0) > 0 else 0,
                "beyond_profit_pattern_hit": 1 if group_hits.get("beyond_profit", 0) > 0 else 0,
                "prompt_match_score": score,
            }
        )
    return pd.DataFrame(rows)


def compute_prompt_match_scores(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = score_prompt_patterns(df, cfg)
    out.insert(0, "chunk_id", df["chunk_id"].to_numpy())
    out["prompt_score"] = out["prompt_match_score"]
    return out
