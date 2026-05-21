import os
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict

# =========================
# Config
# =========================

RAW_DIR = Path("data/raw/edgar/txt")
OUTPUT_PATH = Path("data/clean_2.0/edgar_clean_v4.jsonl")

TARGET_SECTIONS = [
    "item 1",
    "item 1a",
    "item 7"
]

SECTION_MAP = {
    "item 1": "business",
    "item 1a": "risk",
    "item 7": "mdna",
}
# =========================
# Cleaning Utils
# =========================

def clean_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace"""
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.strip()


def remove_boilerplate(text: str) -> str:
    """Remove common SEC boilerplate"""
    patterns = [
        r"forward-looking statements.*?risks and uncertainties",
        r"safe harbor.*?forward-looking statements",
    ]

    text_lower = text.lower()

    for pattern in patterns:
        text_lower = re.sub(pattern, "", text_lower, flags=re.DOTALL)

    return text_lower


def extract_sections(text: str) -> Dict[str, str]:
    """
    Extract key 10-K sections (Item 1, 1A, 7)
    """

    sections = {}

    # Normalize
    text = text.lower()

    # Regex split by Item
    matches = list(re.finditer(r"(item\s+\d+[a]?)", text))

    for i, match in enumerate(matches):
        start = match.start()
        item_name = match.group().strip()

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]

        if any(target in item_name for target in TARGET_SECTIONS):
            sections[item_name] = section_text

    return sections


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into semantic paragraphs
    """
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    return paragraphs


# =========================
# Main Pipeline
# =========================

def process_file(file_path: Path, ticker: str) -> List[Dict]:
    """
    Process one EDGAR file into cleaned chunks
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    # Step 1: clean HTML
    text = clean_html(raw_text)

    # Step 2: remove boilerplate
    text = remove_boilerplate(text)

    # Step 3: extract sections
    sections = extract_sections(text)

    records = []

    year_match = re.search(r"(20\d{2})", file_path.name)
    year = year_match.group(1) if year_match else "unknown"

    for section_name, section_text in sections.items():

        paragraphs = split_into_paragraphs(section_text)

        for i, para in enumerate(paragraphs):

            normalized_section = SECTION_MAP.get(section_name.lower(), "other")

            record = {
                "doc_id": f"{ticker}_{year}_{normalized_section}",
                "chunk_id": f"{ticker}_{year}_{normalized_section}_{i}",
                "company": ticker,
                "ticker": ticker,
                "year": int(year),
                "source": "edgar",
                "source_type": "10k_section",
                "doc_type": normalized_section,
                "section": section_name,
                "page_title": None,
                "url": None,
                "text": para,
                "metadata": {
                "file_name": file_path.name
                }
            }

            records.append(record)

    return records


def run_pipeline():
    all_records = []

    for ticker_dir in RAW_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue

        ticker = ticker_dir.name

        for file in ticker_dir.glob("*full_submission.txt"):

            print(f"Processing {file}...")

            records = process_file(file, ticker)
            all_records.extend(records)

    # Save JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
