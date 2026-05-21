# src/history_consistency/hc_rubric.py

from __future__ import annotations

from typing import Dict, List

from history_consistency.hc_config import (
    DIMENSION_ID,
    DIMENSION_LABEL,
    HC_QUESTION_ID,
    HC_QUESTION_TEXT,
    HC_RUBRIC,
)


def get_hc_question_text() -> str:
    """
    Return the exact HC question text.

    Important:
    Do not modify the wording of HC_QUESTION_TEXT.
    """
    return HC_QUESTION_TEXT


def get_hc_rubric() -> Dict[int, str]:
    """
    Return HC rubric as a score-description dictionary.

    Score range:
    0–5
    """
    return dict(HC_RUBRIC)


def format_hc_rubric_for_prompt() -> str:
    """
    Format the HC rubric into a prompt-ready text block.

    This function preserves the exact rubric descriptions from config.
    """
    lines: List[str] = [
        f"Dimension: {DIMENSION_LABEL}",
        f"Dimension ID: {DIMENSION_ID}",
        f"Question ID: {HC_QUESTION_ID}",
        "",
        "Question:",
        HC_QUESTION_TEXT,
        "",
        "Rubric:",
    ]

    for score in sorted(HC_RUBRIC.keys()):
        lines.append(f"{score}: {HC_RUBRIC[score]}")

    return "\n".join(lines)


def validate_hc_rubric() -> None:
    """
    Validate that HC rubric has exactly six score levels: 0, 1, 2, 3, 4, 5.
    """
    expected = {0, 1, 2, 3, 4, 5}
    actual = set(HC_RUBRIC.keys())

    if actual != expected:
        raise ValueError(
            f"HC rubric must contain exactly scores {expected}, but got {actual}."
        )

    for score, description in HC_RUBRIC.items():
        if not isinstance(score, int):
            raise TypeError(f"Rubric score must be int, got {type(score)}.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Rubric description for score {score} is empty.")


def get_score_description(score: int) -> str:
    """
    Return rubric description for a specific score.
    """
    if score not in HC_RUBRIC:
        raise ValueError(f"Invalid HC score: {score}. Expected 0–5.")
    return HC_RUBRIC[score]


if __name__ == "__main__":
    validate_hc_rubric()
    print(format_hc_rubric_for_prompt())
