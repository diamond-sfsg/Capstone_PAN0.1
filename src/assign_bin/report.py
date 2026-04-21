from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .postprocess import build_basic_stats, build_cross_tab, build_group_counts


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_dict(stats: Dict[str, Any]) -> None:
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:<24}: {v:.4f}")
        else:
            print(f"{k:<24}: {v}")


def _print_series(title: str, series: pd.Series, max_rows: int | None = None) -> None:
    _print_header(title)
    if series.empty:
        print("[EMPTY]")
        return

    if max_rows is not None:
        series = series.head(max_rows)

    print(series.to_string())


def _print_df(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
    _print_header(title)
    if df.empty:
        print("[EMPTY]")
        return

    if max_rows is not None:
        df = df.head(max_rows)

    print(df.to_string())


def print_chunk_report(df: pd.DataFrame) -> None:
    _print_header("PHASE 1 CHUNK NORMALIZE REPORT")

    basic_stats = build_basic_stats(df)
    _print_dict(basic_stats)

    if "source" in df.columns:
        _print_series("COUNTS BY SOURCE", build_group_counts(df, "source"))

    if "quality_flag" in df.columns:
        _print_series("COUNTS BY QUALITY FLAG", build_group_counts(df, "quality_flag"))

    if "is_short_text" in df.columns:
        short_series = (
            df["is_short_text"]
            .fillna(False)
            .map({True: "short", False: "not_short"})
            .value_counts()
        )
        _print_series("SHORT TEXT COUNTS", short_series)

    if "is_duplicate_like" in df.columns:
        dup_series = (
            df["is_duplicate_like"]
            .fillna(False)
            .map({True: "duplicate_like", False: "unique_like"})
            .value_counts()
        )
        _print_series("DUPLICATE COUNTS", dup_series)

    if "token_count" in df.columns and len(df) > 0:
        token_desc = df["token_count"].describe()
        _print_series("TOKEN COUNT DISTRIBUTION", token_desc)

    if "char_count" in df.columns and len(df) > 0:
        char_desc = df["char_count"].describe()
        _print_series("CHAR COUNT DISTRIBUTION", df["char_count"].describe())

    if "source" in df.columns and "quality_flag" in df.columns:
        _print_df(
            "SOURCE x QUALITY FLAG",
            build_cross_tab(df, "source", "quality_flag"),
        )

    if "source" in df.columns and "is_short_text" in df.columns:
        temp = df.copy()
        temp["short_label"] = temp["is_short_text"].fillna(False).map(
            {True: "short", False: "not_short"}
        )
        _print_df(
            "SOURCE x SHORT TEXT",
            build_cross_tab(temp, "source", "short_label"),
        )

    if "source" in df.columns and "is_duplicate_like" in df.columns:
        temp = df.copy()
        temp["dup_label"] = temp["is_duplicate_like"].fillna(False).map(
            {True: "duplicate_like", False: "unique_like"}
        )
        _print_df(
            "SOURCE x DUPLICATE",
            build_cross_tab(temp, "source", "dup_label"),
        )