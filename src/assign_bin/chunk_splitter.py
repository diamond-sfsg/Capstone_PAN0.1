# src/assign_bin/chunk_splitter.py

from __future__ import annotations

from typing import Dict, Any, List, Optional

from .config import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    NORMALIZE_VERSION,
    TARGET_CHUNK_TOKENS,
)
from .normalize import (
    build_chunk_quality_fields,
    clean_text,
    count_tokens,
    split_paragraphs,
    split_sentences,
)


def _safe_str(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_doc_id(record: Dict[str, Any]) -> str:
    existing = _safe_str(record.get("doc_id"))
    if existing:
        return existing

    company = _safe_str(record.get("company")) or "unknown_company"
    year = _safe_str(record.get("year")) or "unknown_year"
    source = _safe_str(record.get("source")) or "unknown_source"
    section = _safe_str(record.get("section")) or "unknown_section"

    section = section.lower().replace(" ", "_")
    return f"{source}_{company}_{year}_{section}"


def _slug(s: str) -> str:
    return (
        _safe_str(s).lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("|", "_")
    )


def make_chunk_id(source: str, company: str, year: Any, doc_id: str, chunk_idx: int) -> str:
    return f"{_slug(source)}|{_slug(company)}|{_slug(year)}|{_slug(doc_id)}|{chunk_idx:04d}"


def _hard_split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """
    最终兜底：任何超长块都强制按 token 数切开
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current).strip())
            current = []

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c.strip()]


def _chunk_by_sentences(paragraph_text: str, max_tokens: int) -> List[str]:
    sentences = split_sentences(paragraph_text)
    if not sentences:
        return _hard_split_by_tokens(paragraph_text, max_tokens)

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        if sent_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_tokens = 0
            chunks.extend(_hard_split_by_tokens(sent, max_tokens))
            continue

        if current_tokens + sent_tokens <= max_tokens:
            current.append(sent)
            current_tokens += sent_tokens
        else:
            if current:
                chunks.append(" ".join(current).strip())
            current = [sent]
            current_tokens = sent_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c.strip()]


def _normalize_paragraph_units(text_raw: str) -> List[str]:
    paragraphs = split_paragraphs(text_raw)
    if not paragraphs:
        cleaned = clean_text(text_raw)
        return [cleaned] if cleaned else []

    units: List[str] = []

    for para in paragraphs:
        para_clean = clean_text(para)
        if not para_clean:
            continue

        para_tokens = count_tokens(para_clean)
        if para_tokens <= MAX_CHUNK_TOKENS:
            units.append(para_clean)
        else:
            units.extend(_chunk_by_sentences(para_clean, MAX_CHUNK_TOKENS))

    return [u for u in units if u.strip()]


def _merge_small_units(
    units: List[str],
    target_tokens: int = TARGET_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> List[str]:
    if not units:
        return []

    merged: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if unit_tokens == 0:
            continue

        if current_tokens == 0:
            current = [unit]
            current_tokens = unit_tokens
            continue

        if current_tokens < min_tokens:
            if current_tokens + unit_tokens <= max_tokens:
                current.append(unit)
                current_tokens += unit_tokens
            else:
                merged.append("\n\n".join(current).strip())
                current = [unit]
                current_tokens = unit_tokens
            continue

        if current_tokens >= target_tokens:
            merged.append("\n\n".join(current).strip())
            current = [unit]
            current_tokens = unit_tokens
            continue

        if current_tokens + unit_tokens <= max_tokens:
            current.append(unit)
            current_tokens += unit_tokens
        else:
            merged.append("\n\n".join(current).strip())
            current = [unit]
            current_tokens = unit_tokens

    if current:
        merged.append("\n\n".join(current).strip())

    return [m for m in merged if m.strip()]


def _final_enforce_max_tokens(chunks: List[str], max_tokens: int) -> List[str]:
    """
    最终硬兜底：任何块只要还超长，就再切一次
    """
    final_chunks: List[str] = []
    for chunk in chunks:
        if count_tokens(chunk) <= max_tokens:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_hard_split_by_tokens(chunk, max_tokens))
    return [c for c in final_chunks if c.strip()]


def split_record_into_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    text_raw = _safe_str(record.get("text_raw"))
    if not text_raw:
        return []

    doc_id = build_doc_id(record)

    units = _normalize_paragraph_units(text_raw)
    merged_chunks = _merge_small_units(units)
    final_chunks = _final_enforce_max_tokens(merged_chunks, MAX_CHUNK_TOKENS)

    chunk_records: List[Dict[str, Any]] = []

    for idx, chunk_text_raw in enumerate(final_chunks, start=1):
        text_clean = clean_text(chunk_text_raw)
        token_count, char_count, is_short_text, quality_flag = build_chunk_quality_fields(text_clean)

        chunk_records.append({
            "chunk_id": make_chunk_id(
                source=record.get("source"),
                company=record.get("company"),
                year=record.get("year"),
                doc_id=doc_id,
                chunk_idx=idx,
            ),
            "doc_id": doc_id,
            "company": record.get("company"),
            "year": record.get("year"),
            "source": record.get("source"),
            "source_file": record.get("source_file"),
            "section": record.get("section"),
            "subsection": record.get("subsection"),
            "text_raw": chunk_text_raw,
            "text_clean": text_clean,
            "token_count": token_count,
            "char_count": char_count,
            "is_short_text": is_short_text,
            "is_exact_duplicate": False,
            "is_same_year_duplicate_like": False,
            "is_cross_year_similar": False,
            "is_duplicate_like": False,
            "duplicate_group": None,
            "similarity_scope": "none",
            "quality_flag": quality_flag,
            "normalize_version": NORMALIZE_VERSION,
        })

    return chunk_records


def split_records_into_chunks(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for record in records:
        all_chunks.extend(split_record_into_chunks(record))
    return all_chunks