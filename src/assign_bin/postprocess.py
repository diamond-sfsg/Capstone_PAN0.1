from __future__ import annotations

import hashlib
from typing import Any, Dict

import pandas as pd

from .normalize import normalize_for_match, normalize_for_similarity
from .config import MAX_CHUNK_TOKENS


# =========================
# 工具函数
# =========================

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _prefix_signature(text: str, max_tokens: int = 80) -> str:
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def _token_set(text: str) -> set[str]:
    return set(text.split())


def jaccard_similarity(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# =========================
# 主逻辑：Similarity Tagging
# =========================

def tag_similarity_relations(
    df: pd.DataFrame,
    near_same_year_threshold: float = 0.85,
    cross_year_threshold: float = 0.80,
    year_gap_limit: int = 5,
    block_on_section: bool = True,
) -> pd.DataFrame:
    """
    目标不是 aggressive dedup，而是打标签：

    - exact_same_year
    - near_same_year
    - cross_year_recurring

    说明：
    - same-year 相似：更偏 redundancy / evidence reuse control
    - cross-year 相似：后续可作为 history consistency 候选信号
    """

    out = df.copy()

    # ---------- 基础校验 ----------
    required_cols = ["chunk_id", "company", "year", "text_clean"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for similarity tagging: {missing}")

    # ---------- 初始化字段 ----------
    for col in [
        "is_exact_duplicate",
        "is_same_year_duplicate_like",
        "is_cross_year_similar",
        "is_duplicate_like",
    ]:
        if col not in out.columns:
            out[col] = False
        else:
            out[col] = out[col].fillna(False).astype(bool)

    if "duplicate_group" not in out.columns:
        out["duplicate_group"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    else:
        out["duplicate_group"] = out["duplicate_group"].astype("object")

    if "similarity_scope" not in out.columns:
        out["similarity_scope"] = pd.Series(["none"] * len(out), index=out.index, dtype="object")
    else:
        out["similarity_scope"] = out["similarity_scope"].fillna("none").astype("object")

    # ---------- 预处理 ----------
    out["_norm_exact"] = out["text_clean"].fillna("").map(normalize_for_match)
    out["_norm_sim"] = out["text_clean"].fillna("").map(normalize_for_similarity)

    out["_exact_hash"] = out["_norm_exact"].map(_hash_text)
    out["_near_sig"] = out["_norm_sim"].map(lambda x: _prefix_signature(x, 80))
    out["_near_hash"] = out["_near_sig"].map(_hash_text)

    # =========================
    # 1) exact same-year duplicate
    # =========================
    same_year_cols = ["company", "year", "_exact_hash"]
    exact_group_sizes = out.groupby(same_year_cols)["chunk_id"].transform("size")
    exact_mask = exact_group_sizes > 1

    out.loc[exact_mask, "is_exact_duplicate"] = True
    out.loc[exact_mask, "is_duplicate_like"] = True
    out.loc[exact_mask, "similarity_scope"] = "exact_same_year"
    out.loc[exact_mask, "duplicate_group"] = out.loc[exact_mask, "_exact_hash"].astype("object")

    # =========================
    # 2) near same-year duplicate
    # =========================
    near_year_cols = ["company", "year", "_near_hash"]
    near_group_sizes = out.groupby(near_year_cols)["chunk_id"].transform("size")
    near_mask = (near_group_sizes > 1) & (~out["is_exact_duplicate"])

    out.loc[near_mask, "is_same_year_duplicate_like"] = True
    out.loc[near_mask, "is_duplicate_like"] = True
    out.loc[near_mask, "similarity_scope"] = "near_same_year"
    out.loc[near_mask, "duplicate_group"] = out.loc[near_mask, "_near_hash"].astype("object")

    # =========================
    # 3) cross-year recurring narrative
    # =========================
    cross_year_map: Dict[str, str] = {}

    # 仅公司内比较，降低复杂度
    for company, g in out.groupby("company", dropna=False):
        g = g[["chunk_id", "year", "_norm_sim", "section"]].copy()
        g = g[g["_norm_sim"].str.len() > 0]

        if len(g) < 2:
            continue

        # 第一层 blocking：prefix signature
        g["_block_key"] = g["_norm_sim"].map(lambda x: _prefix_signature(x, 50))

        for _, sub in g.groupby("_block_key", dropna=False):
            rows = sub.to_dict("records")
            if len(rows) < 2:
                continue

            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a = rows[i]
                    b = rows[j]

                    year_a = a["year"]
                    year_b = b["year"]

                    if pd.isna(year_a) or pd.isna(year_b):
                        continue
                    if year_a == year_b:
                        continue

                    try:
                        if abs(int(year_a) - int(year_b)) > year_gap_limit:
                            continue
                    except Exception:
                        continue

                    if block_on_section:
                        sec_a = _safe_text(a.get("section")).lower()
                        sec_b = _safe_text(b.get("section")).lower()
                        if sec_a and sec_b and sec_a != sec_b:
                            continue

                    sim = jaccard_similarity(a["_norm_sim"], b["_norm_sim"])

                    if sim >= cross_year_threshold:
                        group_key = f"{company}|cross_year|{min(year_a, year_b)}|{max(year_a, year_b)}"
                        cross_year_map[a["chunk_id"]] = group_key
                        cross_year_map[b["chunk_id"]] = group_key

    if cross_year_map:
        cross_mask = out["chunk_id"].isin(cross_year_map.keys())
        out.loc[cross_mask, "is_cross_year_similar"] = True

        # same-year duplicate 优先级更高，不覆盖
        none_mask = cross_mask & (out["similarity_scope"] == "none")
        out.loc[none_mask, "similarity_scope"] = "cross_year_recurring"
        out.loc[none_mask, "duplicate_group"] = (
            out.loc[none_mask, "chunk_id"].map(cross_year_map).astype("object")
        )

    # 清理中间列
    out.drop(
        columns=["_norm_exact", "_norm_sim", "_exact_hash", "_near_sig", "_near_hash"],
        inplace=True,
        errors="ignore",
    )

    return out


# =========================
# 统计函数（给 report.py 用）
# =========================

def build_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)

    short_count = int(df["is_short_text"].fillna(False).sum()) if "is_short_text" in df.columns else 0
    duplicate_count = int(df["is_duplicate_like"].fillna(False).sum()) if "is_duplicate_like" in df.columns else 0
    exact_duplicate_count = int(df["is_exact_duplicate"].fillna(False).sum()) if "is_exact_duplicate" in df.columns else 0
    same_year_duplicate_count = int(df["is_same_year_duplicate_like"].fillna(False).sum()) if "is_same_year_duplicate_like" in df.columns else 0
    cross_year_count = int(df["is_cross_year_similar"].fillna(False).sum()) if "is_cross_year_similar" in df.columns else 0
    garbled_count = int((df["quality_flag"] == "garbled_text").sum()) if "quality_flag" in df.columns else 0
    oversized_count = int((df["token_count"] > MAX_CHUNK_TOKENS).sum()) if "token_count" in df.columns else 0

    stats = {
        "total_chunks": total,
        "short_count": short_count,
        "short_rate": (short_count / total) if total else 0.0,
        "duplicate_count": duplicate_count,
        "duplicate_rate": (duplicate_count / total) if total else 0.0,
        "exact_duplicate_count": exact_duplicate_count,
        "same_year_duplicate_count": same_year_duplicate_count,
        "cross_year_similar_count": cross_year_count,
        "garbled_text_count": garbled_count,
        "oversized_chunk_count": oversized_count,
        "oversized_chunk_rate": (oversized_count / total) if total else 0.0,
    }

    if "token_count" in df.columns and total:
        s = df["token_count"].fillna(0)
        stats.update({
            "token_min": int(s.min()),
            "token_p25": float(s.quantile(0.25)),
            "token_median": float(s.median()),
            "token_mean": float(s.mean()),
            "token_p75": float(s.quantile(0.75)),
            "token_max": int(s.max()),
        })

    return stats


def build_group_counts(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="int64")
    return df[column].fillna("NA").value_counts(dropna=False)


def build_cross_tab(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    if row_col not in df.columns or col_col not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df[row_col].fillna("NA"), df[col_col].fillna("NA"))