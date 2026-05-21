from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PAQuestion:
    question_id: str
    name: str
    question: str
    scoring_type: str  # "evidence" or "evidence_set"
    query: str
    rubric: dict[int, str]


PA_QUESTIONS: dict[str, PAQuestion] = {
    "PA_Q1": PAQuestion(
        question_id="PA_Q1",
        name="Purpose Presence",
        scoring_type="evidence",
        question=(
            "Does the company articulate a purpose statement that clearly specifies the stakeholders it serves, the impact it aims to create, and the reason the company exists?"
        ),
        query=(
            "purpose statement stakeholders serves impact aims to create reason company exists "
            "purpose mission why we exist reason for existence stakeholders intended impact"
        ),
        rubric={
            0: "No purpose statement is present in any corporate document.",
            1: "A vague or implied sense of purpose exists, but it is not explicitly stated.",
            2: "A purpose statement exists but does not clearly specify stakeholders or intended impact.",
            3: "The purpose statement names stakeholders and describes intended impact, but remains somewhat general.",
            4: "The purpose statement clearly identifies stakeholders, impact, and the reason the company exists.",
            5: "The purpose statement is precise, prominent across all major documents, and unambiguously defines who is served, what impact is created, and why the company exists.",
        },
    ),
    "PA_Q2": PAQuestion(
        question_id="PA_Q2",
        name="Clarity",
        scoring_type="evidence",
        question=(
            "Is the purpose articulated in concrete and specific language that describes real outcomes or societal impact, rather than broad aspirational branding?"
        ),
        query=(
            "concrete specific language real outcomes societal impact broad aspirational branding "
            "purpose articulated outcome-oriented measurable real-world outcomes generic branding rhetoric"
        ),
        rubric={
            0: "Language is entirely abstract, aspirational, or indistinguishable from a marketing tagline.",
            1: "Language is mostly aspirational with minimal reference to concrete outcomes or societal impact.",
            2: "Some concrete language exists, but the statement still relies heavily on broad or generic phrasing.",
            3: "The purpose is largely described in specific terms; some aspirational language remains but does not dominate.",
            4: "Language is concrete and outcome-oriented throughout; societal impact is clearly described.",
            5: "Language is precise, measurable, and grounded in real-world outcomes; entirely free of generic branding rhetoric.",
        },
    ),
    "PA_Q3": PAQuestion(
        question_id="PA_Q3",
        name="Distinction from Branding",
        scoring_type="evidence_set",
        question=(
            "Is the purpose positioned within a strategic or operational context, rather than appearing primarily in marketing or promotional materials?"
        ),
        query=(
            "purpose positioned strategic operational context marketing promotional materials "
            "formal documents annual reports sustainability reports strategic priorities operational sections "
            "capital allocation rationale governing business principle"
        ),
        rubric={
            0: (
                "The purpose appears exclusively in marketing or promotional materials (e.g. homepage banners, advertising copy, brand campaigns). It is absent from any strategic or operational documents."
            ),
            1: (
                "The purpose is present in one or two non-marketing documents, but its framing remains indistinguishable from brand messaging. No strategic or operational context is established."
            ),
            2: (
                "The purpose appears in formal documents such as annual reports or sustainability reports, but is confined to introductory or brand-identity sections with no connection to business operations."
            ),
            3: (
                "The purpose is referenced in operational or strategic sections of corporate documents (e.g. risk factors, strategic priorities, capital allocation rationale), though marketing contexts still predominate."
            ),
            4: (
                "The purpose is consistently framed as a strategic and operational guiding principle across multiple document types; its appearance in marketing materials is secondary and clearly distinct."
            ),
            5: (
                "The purpose is explicitly and consistently positioned as a governing business principle across all major corporate documents. Any marketing use is formally distinguished from its strategic role, and this distinction is documented or publicly articulated."
            ),
        },
    ),
}


QUESTION_ORDER = ["PA_Q1", "PA_Q2", "PA_Q3"]


def get_pa_question(question_id: str) -> PAQuestion:
    if question_id not in PA_QUESTIONS:
        raise KeyError(f"Unknown PA question_id: {question_id}")
    return PA_QUESTIONS[question_id]


def format_rubric(question_id: str) -> str:
    question = get_pa_question(question_id)

    lines = [
        f"{question.name}: {question.question}",
        "",
        "Score\tDescription",
    ]

    for score, description in question.rubric.items():
        lines.append(f"{score}\t{description}")

    return "\n".join(lines)


def format_all_rubrics() -> str:
    blocks = []

    for question_id in QUESTION_ORDER:
        blocks.append(format_rubric(question_id))

    return "\n\n".join(blocks)