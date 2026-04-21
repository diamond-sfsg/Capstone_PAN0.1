from __future__ import annotations

PURPOSE_ARTICULATION = {
    "name": "purpose_articulation",
    "description": (
        "Evidence that the company explicitly articulates a purpose statement "
        "describing who it serves, what impact it aims to create, and why it exists. "
        "The articulation should be concrete, stakeholder-aware, and distinct from "
        "pure branding or promotional language."
    ),
    "query_text": (
        "clear articulation of company purpose mission reason for existence "
        "who the company serves stakeholders intended impact societal impact "
        "why the company exists concrete outcome oriented purpose statement "
        "distinct from marketing branding and promotional language strategic business context"
    ),
    # Strong retrieval signals
    "keywords_core": [
        "purpose",
        "mission",
        "our purpose",
        "our mission",
        "reason for existence",
        "why we exist",
        "why the company exists",
        "who we serve",
        "stakeholders",
        "intended impact",
        "societal impact",
        "serve customers",
        "serve communities",
        "serve employees",
        "create impact",
        "make a difference",
    ],
    # Supporting retrieval signals
    "keywords_support": [
        "values",
        "vision",
        "belief",
        "identity",
        "commitment",
        "principles",
        "responsibility",
        "community",
        "customers",
        "employees",
        "society",
        "long term impact",
        "positive impact",
        "outcome",
    ],
    # Signals that may indicate branding-heavy context; keep for future analysis
    "branding_terms": [
        "brand",
        "marketing",
        "campaign",
        "tagline",
        "promotion",
        "promotional",
        "homepage banner",
        "advertising",
    ],
    # Signals suggesting strategic / formal context
    "strategic_context_terms": [
        "strategy",
        "strategic priority",
        "capital allocation",
        "operating model",
        "annual report",
        "investor presentation",
        "sustainability report",
        "governance",
        "business principle",
        "risk factor",
    ],
    "preferred_sections": [
        "mission",
        "purpose",
        "about",
        "about us",
        "our purpose",
        "our mission",
        "values",
        "company overview",
        "letter to shareholders",
        "ceo letter",
        "annual report",
        "sustainability",
        "investor presentation",
        "proxy",
    ],
    "preferred_sources": [
        "official_web",
        "edgar",
    ],
    # Optional report-time labels for the rubric subfacets
    "subsignals": {
        "purpose_presence": [
            "purpose",
            "mission",
            "reason for existence",
            "why we exist",
        ],
        "clarity": [
            "stakeholders",
            "impact",
            "outcome",
            "societal impact",
            "who we serve",
        ],
        "distinction_from_branding": [
            "strategy",
            "operating model",
            "business principle",
            "annual report",
            "investor presentation",
        ],
    },
}

DIMENSIONS = {
    "purpose_articulation": PURPOSE_ARTICULATION,
}