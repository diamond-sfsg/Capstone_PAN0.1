"""Parameterized purpose-driven evaluation scale metadata."""

from __future__ import annotations

DIMENSION_DEFINITIONS = {
    "purpose_articulation": {
        "title": "Purpose Articulation",
        "dimensions": {
            "purpose_presence": {
                "question": "Does the company articulate why it exists, who it serves, and the impact it seeks to create?",
                "score_descriptions": {
                    "0": "No identifiable purpose or mission statement.",
                    "1": "Purpose language exists but reads like a slogan with little substance.",
                    "2": "General values are mentioned, but the reason for existence remains vague.",
                    "3": "A recognizable purpose statement exists, but stakeholders or intended impact are only partly specified.",
                    "4": "The company clearly explains why it exists and what impact it aims to create.",
                    "5": "Purpose is explicit, stakeholder-linked, impact-oriented, and repeated across corporate materials.",
                },
                "evidence_template": {
                    "required_signals": [
                        "explicit purpose statement",
                        "stakeholder mention",
                        "impact or outcome language",
                    ],
                    "preferred_sources": ["web", "linkedin", "sec"],
                },
            },
            "clarity": {
                "question": "Is the purpose concrete, outcome-oriented, and distinguishable from brand language?",
                "score_descriptions": {
                    "0": "Language is ambiguous or operationally meaningless.",
                    "1": "Messaging is primarily promotional or brand-oriented.",
                    "2": "Purpose is partly visible but still abstract.",
                    "3": "Some real outcomes or beneficiaries are described.",
                    "4": "Purpose language is mostly concrete and tied to visible outcomes.",
                    "5": "Purpose clearly connects actions, outcomes, and beneficiaries in specific terms.",
                },
                "evidence_template": {
                    "required_signals": [
                        "observable outcomes",
                        "beneficiary or stakeholder specificity",
                        "low branding dependence",
                    ],
                    "preferred_sources": ["web", "sec"],
                },
            },
            "historical_consistency": {
                "question": "Has the company maintained a consistent purpose narrative across the 2021-2025 window?",
                "score_descriptions": {
                    "0": "Narrative changes frequently or contradicts itself.",
                    "1": "A stable purpose narrative is hard to identify.",
                    "2": "Some recurring themes exist, but drift is significant.",
                    "3": "Core idea persists, though wording or emphasis changes materially.",
                    "4": "Core narrative remains broadly consistent across major disclosures.",
                    "5": "The central purpose remains stable across the full review window.",
                },
                "evidence_template": {
                    "required_signals": [
                        "multi-year evidence",
                        "repeatable narrative",
                        "cross-year consistency",
                    ],
                    "preferred_sources": ["sec"],
                },
            },
            "leadership_centrality": {
                "question": "Do leaders repeatedly frame purpose as a guide for decisions or direction?",
                "score_descriptions": {
                    "0": "Leadership does not reference purpose.",
                    "1": "Leadership mentions purpose rarely and superficially.",
                    "2": "Leadership references purpose, but not as a strategic guide.",
                    "3": "Leadership occasionally links purpose to direction or priorities.",
                    "4": "Leadership frequently uses purpose to frame major initiatives.",
                    "5": "Purpose clearly functions as a leadership and governance principle.",
                },
                "evidence_template": {
                    "required_signals": [
                        "leadership co-mentions",
                        "strategy linkage",
                        "repeat mentions",
                    ],
                    "preferred_sources": ["web", "linkedin", "sec"],
                },
            },
            "distinction_from_branding": {
                "question": "Is purpose embedded in strategy and accountability rather than left as branding?",
                "score_descriptions": {
                    "0": "Purpose is indistinguishable from marketing slogans.",
                    "1": "Messaging is mostly branding language.",
                    "2": "Some strategic linkage exists, but branding still dominates.",
                    "3": "Purpose appears in both messaging and operational contexts.",
                    "4": "Purpose appears in strategy or disclosure materials with clear business relevance.",
                    "5": "Purpose clearly informs strategy, resource allocation, and measurable commitments.",
                },
                "evidence_template": {
                    "required_signals": [
                        "strategy relevance",
                        "measurement or KPI tie-in",
                        "limited brand-only framing",
                    ],
                    "preferred_sources": ["web", "sec"],
                },
            },
        },
    },
    "operational_embedding": {
        "title": "Operational Embedding",
        "dimensions": {
            "corporate_strategy_integration": {
                "question": "Is the stated purpose explicitly integrated into strategic priorities?",
                "score_descriptions": {
                    "0": "No evidence of purpose in strategy.",
                    "1": "Purpose appears only in mission or marketing materials.",
                    "2": "Purpose is mentioned but not tied to strategic priorities.",
                    "3": "Purpose appears in strategy narratives but inconsistently.",
                    "4": "Purpose regularly frames strategic initiatives.",
                    "5": "Purpose clearly guides long-term strategy.",
                },
                "evidence_template": {
                    "required_signals": ["strategic priorities", "purpose linkage", "initiative framing"],
                    "preferred_sources": ["web", "sec"],
                },
            },
            "capital_and_performance_alignment": {
                "question": "Do investments and resource allocation support purpose-related initiatives?",
                "score_descriptions": {
                    "0": "No visible connection between investments and purpose.",
                    "1": "Investments appear driven only by financial logic.",
                    "2": "Indirect support exists but is weakly articulated.",
                    "3": "Several investments align with purpose-related initiatives.",
                    "4": "Major investments frequently align with purpose.",
                    "5": "Capital allocation is consistently justified through purpose and impact goals.",
                },
                "evidence_template": {
                    "required_signals": ["investment language", "purpose linkage", "multi-year support"],
                    "preferred_sources": ["sec", "web"],
                },
            },
            "employee_embedding": {
                "question": "Is purpose reflected in employee programs, incentives, and internal culture?",
                "score_descriptions": {
                    "0": "No evidence of purpose in employee programs.",
                    "1": "Purpose appears only in internal messaging.",
                    "2": "Limited programs connect loosely to purpose.",
                    "3": "Some employee initiatives reflect purpose.",
                    "4": "Purpose is integrated into workforce programs or training.",
                    "5": "Purpose is deeply embedded in employee culture and decision-making.",
                },
                "evidence_template": {
                    "required_signals": ["employee programs", "training or incentives", "purpose linkage"],
                    "preferred_sources": ["linkedin", "web", "sec"],
                },
            },
            "strategic_decision_justification": {
                "question": "Are major strategic moves justified using purpose rather than only growth logic?",
                "score_descriptions": {
                    "0": "No purpose connection in strategic decisions.",
                    "1": "Decisions are justified almost entirely by market or growth logic.",
                    "2": "Occasional purpose references exist.",
                    "3": "Some strategic decisions demonstrate purpose alignment.",
                    "4": "Many strategic initiatives reflect purpose alongside financial goals.",
                    "5": "Purpose is consistently part of the rationale for major decisions.",
                },
                "evidence_template": {
                    "required_signals": ["major initiative", "decision rationale", "purpose linkage"],
                    "preferred_sources": ["web", "sec"],
                },
            },
            "execution_outcome_accountability": {
                "question": "Does the company report measurable progress toward purpose-related outcomes?",
                "score_descriptions": {
                    "0": "No purpose-related outcome reporting.",
                    "1": "General impact claims lack measurable indicators.",
                    "2": "A few metrics exist, but tracking is thin.",
                    "3": "Some measurable KPIs are reported.",
                    "4": "Clear metrics and progress reporting are tied to purpose.",
                    "5": "A comprehensive KPI framework tracks purpose outcomes over time.",
                },
                "evidence_template": {
                    "required_signals": ["metrics", "progress reporting", "purpose or impact target"],
                    "preferred_sources": ["sec", "web"],
                },
            },
        },
    },
    "execution_consistency": {
        "title": "Execution Consistency",
        "dimensions": {
            "corporate_strategy_integration": {
                "question": "Is purpose reflected in company-level strategic priorities and long-term direction?",
                "score_descriptions": {
                    "0": "No evidence of purpose in long-term direction.",
                    "1": "Purpose appears only in high-level messaging.",
                    "2": "Purpose is occasionally connected to strategic priorities.",
                    "3": "Purpose appears in strategic narratives but unevenly.",
                    "4": "Purpose regularly frames major initiatives.",
                    "5": "Purpose clearly guides long-term corporate direction.",
                },
                "evidence_template": {
                    "required_signals": ["strategy continuity", "long-term priority", "purpose linkage"],
                    "preferred_sources": ["sec", "web"],
                },
            },
            "capital_and_performance_alignment": {
                "question": "Do investments and performance drivers reinforce the purpose-related business areas?",
                "score_descriptions": {
                    "0": "No visible alignment.",
                    "1": "Growth drivers appear detached from purpose.",
                    "2": "Some indirect support exists.",
                    "3": "Several major investments align with purpose-related capabilities.",
                    "4": "Investments and growth drivers frequently support purpose-linked capabilities.",
                    "5": "Capital allocation and performance drivers consistently advance intended purpose outcomes.",
                },
                "evidence_template": {
                    "required_signals": ["growth driver", "capability investment", "purpose linkage"],
                    "preferred_sources": ["sec", "web"],
                },
            },
            "execution_consistency": {
                "question": "Do products, launches, and multi-year outcomes show sustained execution of the stated purpose?",
                "score_descriptions": {
                    "0": "No visible execution pattern tied to purpose.",
                    "1": "Purpose is mentioned, but execution evidence is weak.",
                    "2": "Some related initiatives exist but appear isolated.",
                    "3": "Several launches or outcomes align with purpose.",
                    "4": "Purpose-related execution is reflected consistently in products and outcomes.",
                    "5": "A sustained multi-year execution pattern clearly aligns with purpose.",
                },
                "evidence_template": {
                    "required_signals": ["launches or programs", "outcomes", "multi-year continuity"],
                    "preferred_sources": ["web", "sec"],
                },
            },
            "core_capability_reinforcement": {
                "question": "Does the company keep strengthening the core capability most central to its purpose?",
                "score_descriptions": {
                    "0": "No core capability reinforcement is visible.",
                    "1": "A core capability is named but weakly supported.",
                    "2": "Some initiatives support the capability but inconsistently.",
                    "3": "The capability is meaningfully reinforced through products or platforms.",
                    "4": "The company consistently strengthens a purpose-linked core capability.",
                    "5": "The core capability is systematically reinforced across generations of strategy and execution.",
                },
                "evidence_template": {
                    "required_signals": ["platform or capability language", "reinforcement over time", "purpose linkage"],
                    "preferred_sources": ["web", "sec"],
                },
            },
            "industry_impact_breadth": {
                "question": "Are purpose-related capabilities applied across multiple industries or use cases?",
                "score_descriptions": {
                    "0": "No evidence beyond a narrow focus.",
                    "1": "Broader relevance is asserted but weakly evidenced.",
                    "2": "A few scattered applications are visible.",
                    "3": "Multiple credible industry use cases exist.",
                    "4": "Purpose-related capabilities clearly apply across multiple industries or pathways.",
                    "5": "There is broad, sustained multi-industry adoption of purpose-related capabilities.",
                },
                "evidence_template": {
                    "required_signals": ["multiple industries", "credible use cases", "repeat application"],
                    "preferred_sources": ["web", "sec"],
                },
            },
        },
    },
}

EMBEDDING_PROTOTYPES = {
    "purpose_presence": [
        "We exist to serve specific stakeholders and create a defined long-term impact.",
        "Our mission explains who benefits, why the company exists, and what change it seeks.",
    ],
    "clarity": [
        "The company connects concrete actions to observable outcomes and named beneficiaries.",
        "Purpose language is specific, practical, and not just brand aspiration.",
    ],
    "historical_consistency": [
        "The same purpose narrative appears consistently across annual disclosures from 2021 to 2025.",
        "The company repeats a stable purpose narrative across multiple years of corporate reporting.",
    ],
    "leadership_centrality": [
        "Leadership uses purpose as a guide for strategic direction and decisions.",
        "Executives explicitly connect purpose to governance, priorities, or major initiatives.",
    ],
    "distinction_from_branding": [
        "Purpose is embedded in strategy, resource allocation, and measurable commitments.",
        "Purpose language appears in operating and disclosure contexts, not just marketing slogans.",
    ],
    "corporate_strategy_integration": [
        "The company explicitly ties strategy and long-term priorities to its purpose.",
        "Strategic initiatives are framed as advancing the company’s stated purpose.",
    ],
    "capital_and_performance_alignment": [
        "Investments and resource allocation clearly support purpose-related initiatives or capabilities.",
        "The company’s major financial commitments advance its stated impact goals.",
    ],
    "employee_embedding": [
        "Employee programs, culture, and incentives reflect the company’s purpose.",
        "Purpose influences how the workforce is trained, rewarded, or organized.",
    ],
    "strategic_decision_justification": [
        "Major strategic moves are justified partly through purpose, not only growth logic.",
        "The company explains new initiatives using purpose and stakeholder impact language.",
    ],
    "execution_outcome_accountability": [
        "The company reports measurable indicators showing progress toward purpose outcomes.",
        "KPIs and progress reports demonstrate execution against purpose-related commitments.",
    ],
    "execution_consistency": [
        "Products, launches, and business outcomes show a sustained multi-year purpose pattern.",
        "The company consistently translates purpose into execution over time.",
    ],
    "core_capability_reinforcement": [
        "The company keeps strengthening the core capability most central to its purpose.",
        "Platforms, products, and ecosystem development reinforce the central purpose capability.",
    ],
    "industry_impact_breadth": [
        "The company’s purpose-related capabilities are applied across multiple industries or use cases.",
        "Purpose execution is broad enough to show relevance beyond a single product line.",
    ],
}

