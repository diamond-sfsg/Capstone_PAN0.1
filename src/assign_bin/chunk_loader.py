import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

from .config import (
    EDGAR_CLEAN_PATH,
    LINKEDIN_CLEAN_PATH,
    OFFICIAL_WEB_CLEAN_PATH,
    EDGAR_CLEAN_V2_PATH,
    LINKEDIN_CLEAN_V2_PATH,
    OFFICIAL_WEB_CLEAN_V2_PATH,
)


# =========================
# 基础工具
# =========================

def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """安全读取 jsonl"""
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[ERROR] JSON decode failed at {path} line {i}: {e}")
                continue


def _safe_get(d: Dict, keys: List[str], default=None):
    """多字段 fallback"""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# =========================
# 单条 record 标准化
# =========================

def _normalize_record(
    raw: Dict[str, Any],
    source: str,
    source_file: str,
) -> Dict[str, Any]:
    """
    把不同 source 的结构统一成标准 doc-level record
    ⚠️ 不做 chunk 切分
    ⚠️ 不做 text clean
    """

    record = {
        "doc_id": _safe_get(raw, ["doc_id", "id", "document_id"]),
        "company": _safe_get(raw, ["company", "ticker", "org", "name"]),
        "year": _safe_get(raw, ["year", "fiscal_year", "report_year"]),
        "source": source,
        "source_file": source_file,
        "section": _safe_get(raw, ["section", "heading", "title"]),
        "subsection": _safe_get(raw, ["subsection", "subheading"]),
        "text_raw": _safe_get(raw, ["text", "content", "body"]),
    }

    # fallback: 有些数据是段落 list
    if record["text_raw"] is None:
        paragraphs = _safe_get(raw, ["paragraphs", "chunks"])
        if isinstance(paragraphs, list):
            record["text_raw"] = "\n".join(paragraphs)

    return record


# =========================
# 各 source loader
# =========================

def load_edgar_clean(use_v2: bool = False) -> List[Dict[str, Any]]:
    path = EDGAR_CLEAN_V2_PATH if use_v2 else EDGAR_CLEAN_PATH
    records = []

    for raw in _read_jsonl(path):
        rec = _normalize_record(raw, source="edgar", source_file=str(path.name))
        if rec["text_raw"]:
            records.append(rec)

    print(f"[INFO] Loaded EDGAR: {len(records)} docs")
    return records


def load_linkedin_clean(use_v2: bool = False) -> List[Dict[str, Any]]:
    path = LINKEDIN_CLEAN_V2_PATH if use_v2 else LINKEDIN_CLEAN_PATH
    records = []

    for raw in _read_jsonl(path):
        rec = _normalize_record(raw, source="linkedin", source_file=str(path.name))
        if rec["text_raw"]:
            records.append(rec)

    print(f"[INFO] Loaded LinkedIn: {len(records)} docs")
    return records


def load_official_web_clean(use_v2: bool = False) -> List[Dict[str, Any]]:
    path = OFFICIAL_WEB_CLEAN_V2_PATH if use_v2 else OFFICIAL_WEB_CLEAN_PATH
    records = []

    for raw in _read_jsonl(path):
        rec = _normalize_record(raw, source="official_web", source_file=str(path.name))
        if rec["text_raw"]:
            records.append(rec)

    print(f"[INFO] Loaded Official Web: {len(records)} docs")
    return records


# =========================
# 总入口
# =========================

def load_all_sources(use_v2: bool = False) -> List[Dict[str, Any]]:
    """
    统一入口：返回 doc-level records（未 chunk）
    """

    records = []
    records.extend(load_edgar_clean(use_v2=use_v2))
    records.extend(load_linkedin_clean(use_v2=use_v2))
    records.extend(load_official_web_clean(use_v2=use_v2))

    print(f"[INFO] Total loaded records: {len(records)}")
    return records