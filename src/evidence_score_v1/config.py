from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


CONFIG_FILE = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_FILE.parents[2]   # evidence_score_v1 -> src -> project root

DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean_2.0"
PHASE2_DIR = DATA_DIR / "phase2"

DUPLICATE_GROUP_COL = "duplicate_group"
SIMILARITY_SCOPE_COL = "similarity_scope"
YEAR_COL = "year"
COMPANY_COL = "company"
CHUNK_ID_COL = "chunk_id"
TEXT_COL = "text_clean"

DEFAULT_INPUT_CSV = CLEAN_DIR / "unified_chunks_v4.csv"
EVIDENCE_MATRIX_OUTPUT_PATH = PHASE2_DIR / "evidence_score_v1_newdata.csv"
DEFAULT_DIAGNOSTICS_TXT = PHASE2_DIR / "evidence_score_v1_diagnostics_newdata.txt"

HISTORY_BONUS_PER_EXTRA_YEAR = 0.08
HISTORY_BONUS_MAX = 0.40
CROSS_YEAR_SCOPE_VALUE = "cross_year_recurring"

DIMENSION_PREFIX = {
    "purpose_articulation": "pa",
    "history_consistency": "hc",
    "strategy_alignment": "sa",
}


@dataclass(frozen=True)
class TfidfConfig:
    lowercase: bool = True
    stop_words: str = "english"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.9
    sublinear_tf: bool = True
    max_features: int | None = 30000


@dataclass(frozen=True)
class PromptPatternGroup:
    name: str
    patterns: Sequence[str]
    weight: float = 1.0


@dataclass(frozen=True)
class DimensionConfig:
    name: str
    query_text: str
    core_phrases: Sequence[str]
    support_phrases: Sequence[str]
    negative_phrases: Sequence[str]
    preferred_sections: Sequence[str]
    preferred_sources: Sequence[str]
    prompt_pattern_groups: Sequence[PromptPatternGroup] = field(default_factory=tuple)


PURPOSE_ARTICULATION = DimensionConfig(
    name="purpose_articulation",
    query_text=(
        "clear articulation of organizational purpose mission reason for existence "
        "beyond profit stakeholder value long term value values identity"
    ),
    core_phrases=(
        "purpose",
        "our purpose",
        "mission",
        "our mission",
        "why we exist",
        "reason for existence",
        "reason for being",
        "beyond profit",
        "clear purpose",
    ),
    support_phrases=(
        "values",
        "belief",
        "vision",
        "identity",
        "stakeholders",
        "communities",
        "customers",
        "employees",
        "society",
        "long-term value",
    ),
    negative_phrases=(
        "risk factors",
        "forward-looking statements",
        "legal proceedings",
        "table of contents",
        "glossary",
    ),
    preferred_sections=(
        "mission",
        "purpose",
        "about",
        "values",
        "ceo letter",
        "shareholder letter",
        "sustainability",
        "our company",
    ),
    preferred_sources=(
        "official_web",
        "edgar",
        "linkedin",
    ),
    prompt_pattern_groups=(
        PromptPatternGroup(
            name="explicit_purpose",
            patterns=(
                r"\bwe exist to\b",
                r"\bour purpose is to\b",
                r"\bour mission is to\b",
                r"\bwhy we exist\b",
                r"\breason for existence\b",
            ),
            weight=2.0,
        ),
        PromptPatternGroup(
            name="beyond_profit",
            patterns=(
                r"\bbeyond profit\b",
                r"\bmore than profit\b",
                r"\bnot only profit\b",
                r"\bcreate value for society\b",
            ),
            weight=1.5,
        ),
        PromptPatternGroup(
            name="stakeholder_focus",
            patterns=(
                r"\bfor our customers\b",
                r"\bfor communities\b",
                r"\bfor employees\b",
                r"\bfor society\b",
                r"\bfor patients\b",
            ),
            weight=1.25,
        ),
    ),
)


HISTORY_CONSISTENCY = DimensionConfig(
    name="history_consistency",
    query_text=(
        "consistent articulation of purpose and strategy across years continuity over time "
        "enduring mission repeated long term commitment recurring stakeholder oriented "
        "purpose statements and sustained strategic direction"
    ),
    core_phrases=(
        "remain committed",
        "long-term commitment",
        "over time",
        "continue to",
        "continuity",
        "consistent",
        "enduring",
        "for decades",
        "for generations",
        "year after year",
    ),
    support_phrases=(
        "purpose",
        "our purpose",
        "mission",
        "our mission",
        "why we exist",
        "values",
        "identity",
        "strategy",
        "strategic priority",
        "capital allocation",
        "investment",
        "resource allocation",
        "business initiative",
        "long-term strategy",
    ),
    negative_phrases=(
        "risk factors",
        "forward-looking statements",
        "legal proceedings",
        "quarterly results",
        "temporary initiative",
        "table of contents",
    ),
    preferred_sections=(
        "ceo letter",
        "shareholder letter",
        "annual report",
        "about",
        "purpose",
        "values",
        "strategy",
        "sustainability",
        "our company",
    ),
    preferred_sources=(
        "official_web",
        "edgar",
        "linkedin",
    ),
    prompt_pattern_groups=(
        PromptPatternGroup(
            name="continuity_signal",
            patterns=(
                r"\bremain committed to\b",
                r"\bcontinue to\b",
                r"\bover time\b",
                r"\byear after year\b",
                r"\bfor decades\b",
                r"\bfor generations\b",
                r"\blong[- ]term commitment\b",
            ),
            weight=2.0,
        ),
        PromptPatternGroup(
            name="purpose_continuity",
            patterns=(
                r"\bour purpose remains\b",
                r"\bconsistent with our mission\b",
                r"\bwe continue to pursue our mission\b",
                r"\bwe remain committed to our purpose\b",
            ),
            weight=1.75,
        ),
        PromptPatternGroup(
            name="strategy_continuity",
            patterns=(
                r"\blong[- ]term strategy\b",
                r"\bcontinue to invest in\b",
                r"\bsustained investment in\b",
                r"\bconsistent strategic priority\b",
            ),
            weight=1.5,
        ),
    ),
)


STRATEGY_ALIGNMENT = DimensionConfig(
    name="strategy_alignment",
    query_text=(
        "alignment between organizational purpose and strategy capital allocation "
        "investment priorities resource deployment operating priorities business "
        "initiatives execution plans and strategic decisions"
    ),
    core_phrases=(
        "strategy",
        "strategic priority",
        "capital allocation",
        "investment",
        "resource allocation",
        "operating model",
        "business initiative",
        "execution plan",
        "long-term strategy",
        "growth strategy",
    ),
    support_phrases=(
        "priorities",
        "roadmap",
        "transformation",
        "innovation",
        "initiative",
        "program",
        "capabilities",
        "resource deployment",
        "portfolio",
        "expansion",
        "decision-making",
    ),
    negative_phrases=(
        "risk factors",
        "forward-looking statements",
        "legal proceedings",
        "table of contents",
        "glossary",
    ),
    preferred_sections=(
        "strategy",
        "business strategy",
        "capital allocation",
        "management discussion",
        "operating model",
        "innovation",
        "growth",
        "ceo letter",
        "shareholder letter",
    ),
    preferred_sources=(
        "edgar",
        "official_web",
        "linkedin",
    ),
    prompt_pattern_groups=(
        PromptPatternGroup(
            name="purpose_to_strategy_link",
            patterns=(
                r"\bour strategy is to\b",
                r"\bto deliver on our purpose\b",
                r"\bto advance our mission\b",
                r"\baligned with our purpose\b",
                r"\bin support of our mission\b",
            ),
            weight=2.0,
        ),
        PromptPatternGroup(
            name="resource_allocation_signal",
            patterns=(
                r"\bcapital allocation\b",
                r"\binvest in\b",
                r"\binvestment in\b",
                r"\bresource allocation\b",
                r"\bdeploy capital\b",
                r"\ballocate resources\b",
            ),
            weight=1.75,
        ),
        PromptPatternGroup(
            name="execution_priority_signal",
            patterns=(
                r"\bstrategic priority\b",
                r"\boperating priority\b",
                r"\bkey initiative\b",
                r"\blong[- ]term strategy\b",
                r"\bgrowth strategy\b",
                r"\bexecution plan\b",
            ),
            weight=1.5,
        ),
    ),
)


DEFAULT_TFIDF_CONFIG = TfidfConfig()

SUPPORTED_DIMENSIONS: tuple[str, ...] = (
    PURPOSE_ARTICULATION.name,
    HISTORY_CONSISTENCY.name,
    STRATEGY_ALIGNMENT.name,
)

DIMENSION_SCORE_COLUMNS = {
    "purpose_articulation": [
        "pa_lexical_score",
        "pa_tfidf_score",
        "pa_embedding_score",
        "pa_metadata_score",
        "pa_prompt_score",
    ],
    "history_consistency": [
        "hc_lexical_score",
        "hc_tfidf_score",
        "hc_embedding_score",
        "hc_metadata_score",
        "hc_prompt_score",
        "hc_history_bonus_score",
    ],
    "strategy_alignment": [
        "sa_lexical_score",
        "sa_tfidf_score",
        "sa_embedding_score",
        "sa_metadata_score",
        "sa_prompt_score",
    ],
}
