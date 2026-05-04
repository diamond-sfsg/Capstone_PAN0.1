from __future__ import annotations

from .config import (
    DIMENSION_PREFIX,
    HISTORY_CONSISTENCY,
    PURPOSE_ARTICULATION,
    STRATEGY_ALIGNMENT,
)

DIMENSION_REGISTRY = {
    PURPOSE_ARTICULATION.name: PURPOSE_ARTICULATION,
    HISTORY_CONSISTENCY.name: HISTORY_CONSISTENCY,
    STRATEGY_ALIGNMENT.name: STRATEGY_ALIGNMENT,
}


def get_dimension_config(name: str):
    try:
        return DIMENSION_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(DIMENSION_REGISTRY))
        raise KeyError(
            f"Unsupported dimension: {name}. Supported dimensions: {supported}"
        ) from exc


def get_dimension_prefix(name: str) -> str:
    try:
        return DIMENSION_PREFIX[name]
    except KeyError as exc:
        supported = ", ".join(sorted(DIMENSION_PREFIX))
        raise KeyError(
            f"Missing prefix for dimension: {name}. Supported prefixes: {supported}"
        ) from exc


def list_supported_dimensions() -> list[str]:
    return list(DIMENSION_REGISTRY.keys())