# src/strategy_alignment/sa_rubric.py

"""
Rubric definitions for Strategy & Source Alignment.

This rubric should remain faithful to the provided breakdown questions.
The scoring logic is handled outside this file.
"""


SA_RUBRIC = {
    "SA_Q1": {
        "question_id": "SA_Q1",
        "question_name": "R&D & Capital Alignment",
        "question_text": (
            "Does the company's allocation of R&D investment, growth initiatives, "
            "and capital expenditure reflect and reinforce its stated purpose?"
        ),
        "scoring_type": "evidence_level",
        "score_descriptions": {
            0: (
                "No detectable link between stated purpose and resource allocation "
                "decisions."
            ),
            1: (
                "Resource allocation is rarely connected to purpose; alignment "
                "appears coincidental."
            ),
            2: (
                "Some investment areas relate to purpose, but the majority of "
                "capital is directed elsewhere."
            ),
            3: (
                "A meaningful portion of R&D and growth initiatives can be "
                "reasonably linked to the stated purpose."
            ),
            4: (
                "Most major capital and R&D decisions are explicitly aligned with "
                "the stated purpose."
            ),
            5: (
                "Resource allocation is comprehensively and documentably driven by "
                "purpose; explicit justifications are provided in public disclosures."
            ),
        },
    },
    "SA_Q2": {
        "question_id": "SA_Q2",
        "question_name": "Operational Decision-Making",
        "question_text": (
            "Are the company's operational decisions — such as supplier selection, "
            "product development, or market entry — consistently aligned with its "
            "stated purpose?"
        ),
        "scoring_type": "evidence_level",
        "score_descriptions": {
            0: (
                "No evidence that operational decisions are linked to the stated "
                "purpose."
            ),
            1: (
                "Occasional operational decisions reference purpose, but this "
                "appears incidental."
            ),
            2: (
                "Purpose is referenced in some operational areas but is not a "
                "consistent decision-making criterion."
            ),
            3: (
                "Purpose visibly guides operational decisions in several functions "
                "or business units."
            ),
            4: (
                "Operational decisions across most major functions are explicitly "
                "evaluated against the stated purpose."
            ),
            5: (
                "Purpose is formally embedded in operational frameworks; decision "
                "alignment is tracked, reported, and independently verifiable."
            ),
        },
    },
}


SA_RUBRIC_TEXT = """
Strategy & Source Alignment

SA_Q1 — R&D & Capital Alignment:
Does the company's allocation of R&D investment, growth initiatives, and capital expenditure reflect and reinforce its stated purpose?

Score 0: No detectable link between stated purpose and resource allocation decisions.
Score 1: Resource allocation is rarely connected to purpose; alignment appears coincidental.
Score 2: Some investment areas relate to purpose, but the majority of capital is directed elsewhere.
Score 3: A meaningful portion of R&D and growth initiatives can be reasonably linked to the stated purpose.
Score 4: Most major capital and R&D decisions are explicitly aligned with the stated purpose.
Score 5: Resource allocation is comprehensively and documentably driven by purpose; explicit justifications are provided in public disclosures.

SA_Q2 — Operational Decision-Making:
Are the company's operational decisions — such as supplier selection, product development, or market entry — consistently aligned with its stated purpose?

Score 0: No evidence that operational decisions are linked to the stated purpose.
Score 1: Occasional operational decisions reference purpose, but this appears incidental.
Score 2: Purpose is referenced in some operational areas but is not a consistent decision-making criterion.
Score 3: Purpose visibly guides operational decisions in several functions or business units.
Score 4: Operational decisions across most major functions are explicitly evaluated against the stated purpose.
Score 5: Purpose is formally embedded in operational frameworks; decision alignment is tracked, reported, and independently verifiable.
""".strip()


def get_sa_question(question_id: str) -> dict:
    """
    Return one SA rubric question by question_id.
    """
    if question_id not in SA_RUBRIC:
        raise KeyError(f"Unknown SA question_id: {question_id}")
    return SA_RUBRIC[question_id]


def get_sa_rubric_text() -> str:
    """
    Return the full SA rubric as prompt-ready text.
    """
    return SA_RUBRIC_TEXT


def get_score_description(question_id: str, score: int) -> str:
    """
    Return the description for a specific score under a specific SA question.
    """
    question = get_sa_question(question_id)
    descriptions = question["score_descriptions"]

    if score not in descriptions:
        raise KeyError(
            f"Unknown score {score} for question_id {question_id}. "
            "Score must be an integer from 0 to 5."
        )

    return descriptions[score]
