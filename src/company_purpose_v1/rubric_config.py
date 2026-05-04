PURPOSE_RUBRIC = {
    "purpose_articulation": {
        "label": "Purpose Articulation",
        "score_field": "pa_final_score",
        "question": (
            "Does the company clearly articulate a distinct purpose beyond profit?"
        ),
        "criteria": [
            "Clear reason for existence",
            "Beyond short-term financial performance",
            "Specific and company-distinctive language",
            "Authentic purpose articulation rather than generic marketing",
        ],
        "score_guide": {
            5: "Clear, distinctive, authentic purpose beyond profit.",
            4: "Strong purpose articulation with minor generic elements.",
            3: "Some purpose signal, but partially vague or mixed.",
            2: "Weak, generic, or mostly marketing-oriented purpose language.",
            1: "Very weak purpose signal.",
            0: "No usable purpose articulation evidence.",
        },
    },
    "history_consistency": {
        "label": "History Consistency",
        "score_field": "hc_final_score",
        "question": (
            "Is the company's purpose-related narrative consistent over time?"
        ),
        "criteria": [
            "Repeated commitment over multiple years",
            "Long-term framing",
            "Low contradiction between past and current claims",
            "Evidence that purpose is not a one-time rhetorical claim",
        ],
        "score_guide": {
            5: "Strong multi-year consistency with clear continuity.",
            4: "Generally consistent narrative with minor gaps.",
            3: "Some consistency evidence, but limited or uneven.",
            2: "Weak continuity or mostly isolated claims.",
            1: "Very weak consistency signal.",
            0: "No usable history consistency evidence.",
        },
    },
    "strategy_alignment": {
        "label": "Strategy Alignment",
        "score_field": "sa_final_score",
        "question": (
            "Is the stated purpose embedded in strategy, resources, operations, KPIs, or measurable outcomes?"
        ),
        "criteria": [
            "Purpose linked to business strategy",
            "Evidence of resource allocation or investment",
            "Connection to products, operations, or capital priorities",
            "Measurable goals, KPIs, or reported outcomes",
        ],
        "score_guide": {
            5: "Purpose is strongly embedded in strategy and measurable execution.",
            4: "Strong strategic alignment with some measurable support.",
            3: "Moderate alignment, but evidence is incomplete.",
            2: "Weak or mostly rhetorical strategy linkage.",
            1: "Very weak strategy alignment signal.",
            0: "No usable strategy alignment evidence.",
        },
    },
}


DIMENSION_QUERIES = {
    "purpose_articulation": (
        "clear purpose mission reason for existence beyond profit company identity "
        "distinct authentic long-term purpose articulation"
    ),
    "history_consistency": (
        "consistent over time long-term commitment repeated purpose narrative history "
        "continuity across years prior claims current claims"
    ),
    "strategy_alignment": (
        "purpose embedded in strategy resource allocation capital investment KPI "
        "operations measurable outcomes execution business priorities"
    ),
}