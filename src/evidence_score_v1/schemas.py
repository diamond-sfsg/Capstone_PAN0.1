from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    company: str | None
    year: int | str | None
    source: str | None
    section: str | None
    subsection: str | None
    text_raw: str | None
    text_clean: str | None
    duplicate_group: str | None = None
    similarity_scope: str | None = None


@dataclass(frozen=True)
class ScoreFrameSpec:
    dimension_name: str
    prefix: str
    has_history_bonus: bool = False


@dataclass(frozen=True)
class ScoreSummary:
    lexical_score: float = 0.0
    tfidf_score: float = 0.0
    embedding_score: float = 0.0
    metadata_score: float = 0.0
    prompt_score: float = 0.0
    history_bonus_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lexical_score": self.lexical_score,
            "tfidf_score": self.tfidf_score,
            "embedding_score": self.embedding_score,
            "metadata_score": self.metadata_score,
            "prompt_score": self.prompt_score,
            "history_bonus_score": self.history_bonus_score,
        }