"""Aggregate scoring logic."""

from __future__ import annotations

from config.scoring_config import DEFAULT_SCORE_WEIGHTS


def compute_aggregate_score(scores):
    """Combine individual scores into a single aggregate value."""
    aggregate = 0.0
    for key, weight in DEFAULT_SCORE_WEIGHTS.items():
        aggregate += scores[key]["overall"] * weight
    aggregate = round(aggregate, 2)
    if aggregate >= 4.0:
        label = "strongly purpose-driven"
    elif aggregate >= 3.0:
        label = "moderately purpose-driven"
    elif aggregate >= 2.0:
        label = "emerging or mixed"
    else:
        label = "weakly evidenced"
    return {"overall": aggregate, "label": label}


def apply_llm_adjustment(aggregate_score, llm_review):
    """Apply a bounded LLM delta after heuristic scoring."""
    if not llm_review or not llm_review.get("enabled"):
        return aggregate_score
    review = llm_review.get("llm_review")
    if not review:
        return aggregate_score
    delta = float(review.get("overall_adjustment", {}).get("delta", 0.0))
    delta = max(-0.5, min(0.5, delta))
    adjusted = round(max(0.0, min(5.0, aggregate_score["overall"] + delta)), 2)
    if adjusted >= 4.0:
        label = "strongly purpose-driven"
    elif adjusted >= 3.0:
        label = "moderately purpose-driven"
    elif adjusted >= 2.0:
        label = "emerging or mixed"
    else:
        label = "weakly evidenced"
    result = dict(aggregate_score)
    result["heuristic_overall"] = aggregate_score["overall"]
    result["overall"] = adjusted
    result["label"] = label
    result["llm_delta"] = delta
    return result
