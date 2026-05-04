from __future__ import annotations

from typing import Iterable

import pandas as pd

from .embedding_match import compute_embedding_scores
from .history_bonus import compute_history_bonus
from .lexical_match import compute_lexical_scores
from .metadata_match import compute_metadata_scores
from .prompt_match import compute_prompt_match_scores
from .tfidf_match import compute_tfidf_scores


STANDARD_BASE_SCORE_COLUMNS = [
    "lexical_score",
    "tfidf_score",
    "embedding_score",
    "metadata_score",
    "prompt_score",
]


def _select_and_rename_score_columns(
    df: pd.DataFrame,
    source_candidates: list[str],
    target_name: str,
) -> pd.DataFrame:
    """
    Pick the first available source column from source_candidates
    and rename it to target_name.
    """
    for col in source_candidates:
        if col in df.columns:
            return df[["chunk_id", col]].rename(columns={col: target_name})
    raise KeyError(
        f"None of the candidate columns {source_candidates} found in dataframe columns: {list(df.columns)}"
    )


def _build_standard_score_frame(
    lexical_df: pd.DataFrame,
    tfidf_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize scorer outputs into a single standard frame with:
    chunk_id, lexical_score, tfidf_score, embedding_score, metadata_score, prompt_score
    """
    lexical_std = _select_and_rename_score_columns(
        lexical_df,
        source_candidates=["lexical_score", "lexical_raw_score"],
        target_name="lexical_score",
    )
    tfidf_std = _select_and_rename_score_columns(
        tfidf_df,
        source_candidates=["tfidf_score", "tfidf_cosine"],
        target_name="tfidf_score",
    )
    embedding_std = _select_and_rename_score_columns(
        embedding_df,
        source_candidates=["embedding_score", "embedding_cosine"],
        target_name="embedding_score",
    )
    metadata_std = _select_and_rename_score_columns(
        metadata_df,
        source_candidates=["metadata_score", "metadata_total_score"],
        target_name="metadata_score",
    )
    prompt_std = _select_and_rename_score_columns(
        prompt_df,
        source_candidates=["prompt_score", "prompt_match_score"],
        target_name="prompt_score",
    )

    merged = lexical_std.merge(tfidf_std, on="chunk_id", how="outer")
    merged = merged.merge(embedding_std, on="chunk_id", how="outer")
    merged = merged.merge(metadata_std, on="chunk_id", how="outer")
    merged = merged.merge(prompt_std, on="chunk_id", how="outer")

    for col in STANDARD_BASE_SCORE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    return merged


def _prefix_score_columns(
    df: pd.DataFrame,
    prefix: str,
    include_history_bonus: bool = False,
) -> pd.DataFrame:
    """
    Rename standard score columns into prefixed output columns.
    """
    rename_map = {
        "lexical_score": f"{prefix}_lexical_score",
        "tfidf_score": f"{prefix}_tfidf_score",
        "embedding_score": f"{prefix}_embedding_score",
        "metadata_score": f"{prefix}_metadata_score",
        "prompt_score": f"{prefix}_prompt_score",
    }

    if include_history_bonus and "history_bonus_score" in df.columns:
        rename_map["history_bonus_score"] = f"{prefix}_history_bonus_score"

    keep_cols = ["chunk_id"] + list(rename_map.keys())
    return df[keep_cols].rename(columns=rename_map)


def _add_sum_score(
    df: pd.DataFrame,
    prefix: str,
    include_history_bonus: bool = False,
) -> pd.DataFrame:
    """
    Add unweighted sum score for one dimension.
    """
    score_cols = [
        f"{prefix}_lexical_score",
        f"{prefix}_tfidf_score",
        f"{prefix}_embedding_score",
        f"{prefix}_metadata_score",
        f"{prefix}_prompt_score",
    ]
    if include_history_bonus:
        score_cols.append(f"{prefix}_history_bonus_score")

    for col in score_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df[f"{prefix}_sum_score"] = df[score_cols].sum(axis=1)
    return df


def compute_dimension_score_frame(
    df: pd.DataFrame,
    dim_cfg,
    prefix: str,
    include_history_bonus: bool = False,
) -> pd.DataFrame:
    """
    Compute one dimension's standard score frame and return prefixed columns.

    Returns:
        chunk_id
        <prefix>_lexical_score
        <prefix>_tfidf_score
        <prefix>_embedding_score
        <prefix>_metadata_score
        <prefix>_prompt_score
        [<prefix>_history_bonus_score]
        <prefix>_sum_score
    """
    lexical_df = compute_lexical_scores(df, dim_cfg)
    tfidf_df = compute_tfidf_scores(df, dim_cfg)
    embedding_df = compute_embedding_scores(df, dim_cfg)
    metadata_df = compute_metadata_scores(df, dim_cfg)
    prompt_df = compute_prompt_match_scores(df, dim_cfg)

    standard_df = _build_standard_score_frame(
        lexical_df=lexical_df,
        tfidf_df=tfidf_df,
        embedding_df=embedding_df,
        metadata_df=metadata_df,
        prompt_df=prompt_df,
    )

    if include_history_bonus:
        history_df = compute_history_bonus(
            df[["chunk_id", "company", "year", "duplicate_group", "similarity_scope"]]
        )
        history_df = _select_and_rename_score_columns(
            history_df,
            source_candidates=["history_bonus_score"],
            target_name="history_bonus_score",
        )
        standard_df = standard_df.merge(history_df, on="chunk_id", how="left")
        standard_df["history_bonus_score"] = (
            pd.to_numeric(standard_df["history_bonus_score"], errors="coerce").fillna(0.0)
        )

    prefixed = _prefix_score_columns(
        standard_df,
        prefix=prefix,
        include_history_bonus=include_history_bonus,
    )
    prefixed = _add_sum_score(
        prefixed,
        prefix=prefix,
        include_history_bonus=include_history_bonus,
    )
    return prefixed


def merge_dimension_frames(
    base_df: pd.DataFrame,
    score_frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge multiple prefixed dimension score frames back to the base chunk dataframe.
    """
    out = base_df.copy()
    for frame in score_frames:
        out = out.merge(frame, on="chunk_id", how="left")
    return out