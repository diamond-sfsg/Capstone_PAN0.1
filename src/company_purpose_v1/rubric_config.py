PURPOSE_RUBRIC = {
    "purpose_articulation": {
        "label": "Purpose Articulation",
        "score_field": "pa_final_score",
        "questions": [
            {
                "label": "Purpose Articulation Q1",
                "score_field": "pa_Q1",
                "question": (
                    "Purpose Presence: Does the company articulate a purpose statement that clearly specifies the stakeholders it serves, the impact it aims to create, and the reason the company exists?"
                ),
                "score_guide": {
                    5: "The purpose statement is precise, prominent across all major documents, and unambiguously defines who is served, what impact is created, and why the company exists.",
                    4: "The purpose statement clearly identifies stakeholders, impact, and the reason the company exists.",
                    3: "The purpose statement names stakeholders and describes intended impact, but remains somewhat general.",
                    2: "A purpose statement exists but does not clearly specify stakeholders or intended impact.",
                    1: "A vague or implied sense of purpose exists, but it is not explicitly stated.",
                    0: "No purpose statement is present in any corporate document.",
                },
            },
            {
                "label": "Purpose Articulation Q2",
                "score_field": "pa_Q2",
                "question": (
                    "Clarity: Is the purpose articulated in concrete and specific language that describes real outcomes or societal impact, rather than broad aspirational branding?"
                ),
                "score_guide": {
                    5: "Language is precise, measurable, and grounded in real-world outcomes; entirely free of generic branding rhetoric.",
                    4: "Language is concrete and outcome-oriented throughout; societal impact is clearly described.",
                    3: "The purpose is largely described in specific terms; some aspirational language remains but does not dominate.",
                    2: "Some concrete language exists, but the statement still relies heavily on broad or generic phrasing.",
                    1: "Language is mostly aspirational with minimal reference to concrete outcomes or societal impact.",
                    0: "Language is entirely abstract, aspirational, or indistinguishable from a marketing tagline.",
                },
            },
            {
                "label": "Purpose Articulation Q3",
                "score_field": "pa_Q3",
                "question": (
                    "Distinction from Branding: Is the purpose positioned within a strategic or operational context, rather than appearing primarily in marketing or promotional materials?"
                ),
                "score_guide": {
                    5: "The purpose is explicitly and consistently positioned as a governing business principle across all major corporate documents. Any marketing use is formally distinguished from its strategic role, and this distinction is documented or publicly articulated.",
                    4: "The purpose is consistently framed as a strategic and operational guiding principle across multiple document types; its appearance in marketing materials is secondary and clearly distinct.",
                    3: "The purpose is referenced in operational or strategic sections of corporate documents (e.g. risk factors, strategic priorities, capital allocation rationale), though marketing contexts still predominate.",
                    2: "The purpose appears in formal documents such as annual reports or sustainability reports, but is confined to introductory or brand-identity sections with no connection to business operations.",
                    1: "The purpose is present in one or two non-marketing documents, but its framing remains indistinguishable from brand messaging. No strategic or operational context is established.",
                    0: "The purpose appears exclusively in marketing or promotional materials (e.g. homepage banners, advertising copy, brand campaigns). It is absent from any strategic or operational documents.",
                },
            },
        ],
    },
    "history_consistency": {
        "label": "History Consistency",
        "score_field": "hc_final_score",
        "questions": [
            {
                "label": "History Consistency Q1",
                "score_field": "hc_Q1",
                "question": (
                    "Historical Consistency: To what extent has the company maintained a consistent purpose narrative over the past ten years across annual reports, CEO letters, and corporate disclosures?"
                ),
                "criteria": [
                    "Repeated commitment over multiple years",
                    "Long-term framing",
                    "Low contradiction between past and current claims",
                    "Evidence that purpose is not a one-time rhetorical claim",
                ],
                "score_guide": {
                    5: "Core narrative remains almost unchanged across major disclosures, or any changes are clearly explainable without altering the central purpose.",
                    4: "Core narrative remains largely consistent across major disclosures.",
                    3: "The core idea remains consistent, but wording differs significantly across documents.",
                    2: "Some recurring themes exist, but the narrative frequently drifts.",
                    1: "Core theme shifts often, making it difficult to identify a consistent narrative.",
                    0: "Narrative frequently changes or contains contradictions.",
                },
            },
        ],
    },
    "strategy_alignment": {
        "label": "Strategy Alignment",
        "score_field": "sa_final_score",
        "questions": [
            {
                "label": "Strategy Alignment Q1",
                "score_field": "sa_Q1",
                "question": (
                    "R&D & Capital Alignment: Does the company's allocation of R&D investment, growth initiatives, and capital expenditure reflect and reinforce its stated purpose?"
                ),
                "criteria": [
                    "Purpose linked to business strategy",
                    "Evidence of resource allocation or investment",
                    "Connection to products, operations, or capital priorities",
                    "Measurable goals, KPIs, or reported outcomes",
                ],
                "score_guide": {
                    5: "Resource allocation is comprehensively and documentably driven by purpose; explicit justifications are provided in public disclosures.",
                    4: "Most major capital and R&D decisions are explicitly aligned with the stated purpose.",
                    3: "A meaningful portion of R&D and growth initiatives can be reasonably linked to the stated purpose.",
                    2: "Some investment areas relate to purpose, but the majority of capital is directed elsewhere.",
                    1: "Resource allocation is rarely connected to purpose; alignment appears coincidental.",
                    0: "No detectable link between stated purpose and resource allocation decisions.",
                },
            },
            {
                "label": "Strategy Alignment Q2",
                "score_field": "sa_Q2",
                "question": (
                    "Operational Decision-Making: Are the company's operational decisions, such as supplier selection, product development, or market entry, consistently aligned with its stated purpose?"
                ),
                "criteria": [
                    "Purpose linked to operational decisions",
                    "Evidence from supplier selection, product development, market entry, or similar choices",
                    "Connection to products, operations, or capital priorities",
                    "Measurable goals, KPIs, or reported outcomes",
                ],
                "score_guide": {
                    5: "Purpose is formally embedded in operational frameworks; decision alignment is tracked, reported, and independently verifiable.",
                    4: "Operational decisions across most major functions are explicitly evaluated against the stated purpose.",
                    3: "Purpose visibly guides operational decisions in several functions or business units.",
                    2: "Purpose is referenced in some operational areas but is not a consistent decision-making criterion.",
                    1: "Occasional operational decisions reference purpose, but this appears incidental.",
                    0: "No evidence that operational decisions are linked to the stated purpose.",
                },
            },
        ],
    },
}


DIMENSION_QUERIES = {
    "purpose_articulation": (
        "clear purpose mission reason for existence beyond profit company identity "
        "distinct authentic long-term purpose articulation stakeholders impact outcomes"
    ),
    "history_consistency": (
        "consistent over time long-term commitment repeated purpose narrative history "
        "continuity across years prior claims current claims"
    ),
    "strategy_alignment": (
        "purpose embedded in strategy resource allocation capital investment KPI "
        "operations measurable outcomes execution business priorities supplier product market"
    ),
}
