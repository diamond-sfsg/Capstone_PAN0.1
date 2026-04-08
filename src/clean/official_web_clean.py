import os
import re
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd


RAW_DIR = Path("data/raw/official_web")
OUTPUT_PATH = Path("data/clean/official_web_clean.jsonl")


KEEP_KEYWORDS = [
    "about", "mission", "purpose", "values",
    "sustainability", "impact", "esg",
    "governance", "leadership",
    "strategy", "investor", "annual", "report",
    "responsibility", "community", "planet"
]

DROP_KEYWORDS = [
    "product", "pricing", "shop",
    "careers", "jobs", "event",
    "privacy", "terms", "cookie",
    "login", "signup",
    "support", "faq",
    "newsletter", "subscribe",
    "site-map", "sitemap"
]


def clean_text(text: str) -> str:
    if not text or str(text).lower() == "nan":
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    # common website noise
    noise_patterns = [
        r"related tags share",
        r"trending news and stories",
        r"read more",
        r"learn more",
        r"you might also like",
        r"more news from around the world",
        r"page overview",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_relevant_page(url: str, title: str, meta_description: str, text: str) -> bool:
    joined = " ".join([
        str(url or ""),
        str(title or ""),
        str(meta_description or ""),
        str(text or "")[:1000],   # only first part for cheap filtering
    ]).lower()

    if any(k in joined for k in DROP_KEYWORDS):
        return False

    if any(k in joined for k in KEEP_KEYWORDS):
        return True

    return False


def classify_doc_type(url: str, title: str, meta_description: str) -> str:
    joined = " ".join([
        str(url or ""),
        str(title or ""),
        str(meta_description or "")
    ]).lower()

    if "mission" in joined or "purpose" in joined:
        return "mission_page"
    if "sustainability" in joined or "esg" in joined or "planet" in joined:
        return "sustainability_page"
    if "governance" in joined:
        return "governance_page"
    if "leadership" in joined:
        return "leadership_page"
    if "investor" in joined or "annual" in joined or "report" in joined:
        return "investor_page"
    if "impact" in joined or "community" in joined or "responsibility" in joined:
        return "impact_page"
    if "about" in joined or "values" in joined:
        return "about_page"

    return "general_page"


def split_semantic_chunks(text: str, min_len: int = 300, max_len: int = 1600) -> List[str]:
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []

    for sent in sentences:
        candidate = " ".join(current + [sent]).strip()

        if len(candidate) <= max_len:
            current.append(sent)
        else:
            if current:
                chunk = " ".join(current).strip()
                if len(chunk) >= min_len:
                    chunks.append(chunk)
            current = [sent]

    if current:
        chunk = " ".join(current).strip()
        if len(chunk) >= min_len:
            chunks.append(chunk)

    if not chunks and len(text) >= min_len:
        chunks = [text[:max_len]]

    return chunks


def read_pages_table(company_dir: Path) -> pd.DataFrame:
    csv_path = company_dir / "pages.csv"
    jsonl_path = company_dir / "pages.jsonl"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if jsonl_path.exists():
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    return pd.DataFrame()


def process_company(company_dir: Path) -> List[Dict]:
    records = []
    company = company_dir.name

    df = read_pages_table(company_dir)
    if df.empty:
        print(f"[WARN] No pages table found for {company}")
        return records

    print(f"[INFO] {company}: {len(df)} raw pages")

    kept_pages = 0

    for idx, row in df.iterrows():
        url = row.get("url", None)
        title = row.get("title", None)
        meta_description = row.get("meta_description", None)
        path = row.get("path", None)
        text = row.get("text", None)

        cleaned = clean_text(text)

        if len(cleaned) < 200:
            continue

        # optional: trust precomputed relevance if present
        if "is_probably_relevant" in df.columns:
            preflag = row.get("is_probably_relevant", None)
            if str(preflag).lower() == "false":
                continue

        if not is_relevant_page(url, title, meta_description, cleaned):
            continue

        kept_pages += 1
        doc_type = classify_doc_type(url, title, meta_description)
        page_title = None if pd.isna(title) else str(title)

        chunks = split_semantic_chunks(cleaned)

        for i, chunk in enumerate(chunks):
            record = {
                "doc_id": f"{company}_{idx}",
                "chunk_id": f"{company}_{idx}_{i}",
                "company": company,
                "ticker": None,
                "year": None,
                "source": "official_web",
                "source_type": "website_page",
                "doc_type": doc_type,
                "section": None,
                "page_title": page_title,
                "url": None if pd.isna(url) else str(url),
                "text": chunk,
                "metadata": {
                    "path": None if pd.isna(path) else str(path),
                    "meta_description": None if pd.isna(meta_description) else str(meta_description),
                    "text_length_raw": int(row.get("text_length", 0)) if "text_length" in df.columns and pd.notna(row.get("text_length", None)) else None,
                    "purpose_signal_count": int(row.get("purpose_signal_count", 0)) if "purpose_signal_count" in df.columns and pd.notna(row.get("purpose_signal_count", None)) else None,
                }
            }
            records.append(record)

    print(f"[INFO] {company}: kept {kept_pages} pages -> {len(records)} chunks")
    return records


def run_pipeline():
    all_records = []

    for company_dir in RAW_DIR.iterdir():
        if not company_dir.is_dir():
            continue

        company_records = process_company(company_dir)
        all_records.extend(company_records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved {len(all_records)} cleaned records to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()