from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ENABLE_EMBEDDING


def compute_embedding_scores(
    df: pd.DataFrame,
    query_text: str,
    text_col: str = "text_clean",
) -> pd.DataFrame:
    """
    Scaffold version.

    If ENABLE_EMBEDDING is False, return 0.0 for all rows.
    Later this module can be upgraded to call an embedding API or local model.
    """
    out = df.copy()

    if not ENABLE_EMBEDDING:
        out["embedding_score"] = 0.0
        return out

    # Placeholder for future implementation
    out["embedding_score"] = 0.0
    return out