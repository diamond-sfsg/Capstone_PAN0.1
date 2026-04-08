import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw/linkedin")
OUTPUT_PATH = Path("data/clean/linkedin_clean.jsonl")


def normalize_company_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", " ")
    return text.strip()


def deduplicate_lines(text: str) -> str:
    lines = [line.strip() for line in re.split(r"[.\n]+", text) if line.strip()]
    seen = set()
    unique = []
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            unique.append(line)
    return ". ".join(unique)


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = clean_text(text)

    # remove common LinkedIn UI noise
    noise_patterns = [
        r"join now",
        r"sign in",
        r"skip to main content",
        r"linkedin",
        r"agree & join linkedin",
        r"cookie",
        r"privacy policy",
        r"user agreement",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = clean_text(text)
    text = deduplicate_lines(text)
    return text


def split_semantic_chunks(text: str, min_len: int = 120, max_len: int = 1200) -> List[str]:
    """
    Split text into paragraph-like semantic chunks.
    For LinkedIn company pages, usually 1-3 chunks is enough.
    """
    if not text:
        return []

    # first split by sentence-ish boundaries
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks = []
    current = []

    for part in parts:
        candidate = " ".join(current + [part]).strip()
        if len(candidate) <= max_len:
            current.append(part)
        else:
            if current:
                chunk = " ".join(current).strip()
                if len(chunk) >= min_len:
                    chunks.append(chunk)
            current = [part]

    if current:
        chunk = " ".join(current).strip()
        if len(chunk) >= min_len:
            chunks.append(chunk)

    # fallback: if nothing survives and text exists
    if not chunks and len(text) >= min_len:
        chunks = [text[:max_len]]

    return chunks


def read_pages_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None


def extract_structured_text(row: pd.Series) -> Dict[str, Optional[str]]:
    title = clean_text(str(row.get("title", "") or ""))
    meta_description = clean_text(str(row.get("meta_description", "") or ""))
    og_title = clean_text(str(row.get("og_title", "") or ""))
    og_description = clean_text(str(row.get("og_description", "") or ""))
    final_url = clean_text(str(row.get("final_url", "") or ""))
    canonical_url = clean_text(str(row.get("canonical_url", "") or ""))

    fields = [
        ("title", title),
        ("meta_description", meta_description),
        ("og_title", og_title),
        ("og_description", og_description),
    ]

    text_blocks = []
    for label, value in fields:
        if value and value.lower() != "nan":
            text_blocks.append(f"{label}: {value}")

    structured_text = "\n".join(text_blocks).strip()

    return {
        "structured_text": structured_text,
        "title": title if title.lower() != "nan" else None,
        "url": final_url or canonical_url or None,
        "meta_description": meta_description if meta_description.lower() != "nan" else None,
        "og_title": og_title if og_title.lower() != "nan" else None,
        "og_description": og_description if og_description.lower() != "nan" else None,
    }


def read_html_text(html_dir: Path) -> str:
    if not html_dir.exists():
        return ""

    html_files = list(html_dir.glob("*.html"))
    if not html_files:
        return ""

    collected = []
    for html_file in html_files:
        try:
            raw_html = html_file.read_text(encoding="utf-8", errors="ignore")
            txt = clean_html_to_text(raw_html)
            if txt:
                collected.append(txt)
        except Exception as e:
            print(f"[WARN] Failed to parse {html_file}: {e}")

    merged = "\n".join(collected)
    return clean_text(merged)


def build_records_for_company(company_dir: Path) -> List[Dict]:
    records = []
    company = normalize_company_name(company_dir.name)

    pages_csv = company_dir / "pages.csv"
    raw_html_dir = company_dir / "raw_html"

    df = read_pages_csv(pages_csv)
    html_text = read_html_text(raw_html_dir)

    row_data = {}
    if df is not None and len(df) > 0:
        row_data = extract_structured_text(df.iloc[0])

    structured_text = row_data.get("structured_text", "") if row_data else ""
    page_title = row_data.get("title") if row_data else None
    url = row_data.get("url") if row_data else None

    combined_parts = []
    if structured_text:
        combined_parts.append(structured_text)
    if html_text:
        combined_parts.append(html_text)

    combined_text = "\n".join(combined_parts).strip()
    combined_text = clean_text(combined_text)

    if not combined_text:
        return records

    chunks = split_semantic_chunks(combined_text)

    for i, chunk in enumerate(chunks):
        record = {
            "doc_id": f"{company}_linkedin_company_page",
            "chunk_id": f"{company}_linkedin_company_page_{i}",
            "company": company,
            "ticker": None,
            "year": None,
            "source": "linkedin",
            "source_type": "linkedin_company_page",
            "doc_type": "company_about",
            "section": None,
            "page_title": page_title,
            "url": url,
            "text": chunk,
            "metadata": {
                "company_dir": company_dir.name,
                "has_pages_csv": pages_csv.exists(),
                "has_raw_html": raw_html_dir.exists(),
            },
        }
        records.append(record)

    return records


def run_pipeline():
    all_records = []

    for company_dir in RAW_DIR.iterdir():
        if not company_dir.is_dir():
            continue

        print(f"[INFO] Processing LinkedIn company dir: {company_dir.name}")
        records = build_records_for_company(company_dir)
        all_records.extend(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved {len(all_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()