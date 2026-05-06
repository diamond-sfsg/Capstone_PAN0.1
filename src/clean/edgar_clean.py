import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


# =========================
# Config
# =========================

RAW_DIR = Path("data/edgar_by_type")
OUTPUT_PATH = Path("data/clean/edgar_clean.jsonl")

TARGET_SECTIONS = [
    "item 1",
    "item 1a",
    "item 7",
]

SECTION_MAP = {
    "item 1": "business",
    "item 1a": "risk",
    "item 7": "mdna",
}

SECTION_HEADERS = {
    "item 1": [
        "business",
    ],
    "item 1a": [
        "risk factors",
        "the business, financial condition and operating results",
    ],
    "item 7": [
        "management's discussion and analysis of financial condition and results of operations",
        "managements discussion and analysis of financial condition and results of operations",
        "management discussion and analysis of financial condition and results of operations",
    ],
}


# =========================
# Cleaning Utils
# =========================

def clean_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace while preserving line breaks."""
    soup = BeautifulSoup(text, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    extracted = soup.get_text(separator="\n")
    return clean_text(extracted)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    patterns = [
        r"forward-looking statements.*?risks and uncertainties",
        r"safe harbor.*?forward-looking statements",
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    return clean_text(cleaned)


def normalize_heading(text: str) -> str:
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections_by_item(text: str) -> Dict[str, str]:
    sections = {}
    normalized = normalize_heading(text)
    matches = list(re.finditer(r"\bitem\s+(\d+[a-z]?)\b", normalized))

    for i, match in enumerate(matches):
        item_name = f"item {match.group(1)}"
        if item_name not in TARGET_SECTIONS:
            continue

        start = match.start()
        end = len(normalized)
        for next_match in matches[i + 1:]:
            next_item = f"item {next_match.group(1)}"
            if next_item != item_name:
                end = next_match.start()
                break

        section_text = normalized[start:end].strip()
        if len(section_text) > 300:
            sections[item_name] = section_text

    return sections


def find_first_line(lines: List[str], needles: List[str], start: int = 0) -> Optional[int]:
    for idx in range(start, len(lines)):
        normalized = normalize_heading(lines[idx])
        if any(needle in normalized for needle in needles):
            return idx
    return None


def extract_sections_by_headers(text: str) -> Dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {}

    starts: List[Tuple[str, int]] = []
    for item_name in TARGET_SECTIONS:
        idx = find_first_line(lines, SECTION_HEADERS[item_name])
        if idx is not None:
            starts.append((item_name, idx))

    starts = sorted(set(starts), key=lambda x: x[1])
    if not starts:
        return {}

    sections = {}
    for i, (item_name, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()
        if len(section_text) > 300:
            sections[item_name] = clean_text(section_text)

    return sections


def extract_sections(text: str) -> Dict[str, str]:
    sections = extract_sections_by_item(text)
    if sections:
        return sections

    sections = extract_sections_by_headers(text)
    if sections:
        return sections

    return {"10k_full_text": text}


def read_metadata(metadata_path: Path) -> Dict:
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to read metadata {metadata_path}: {exc}")
        return {}


def find_metadata_path(file_path: Path) -> Path:
    relative = file_path.relative_to(RAW_DIR / file_path.parts[-3])
    ticker = relative.parts[0]
    stem = file_path.stem.replace("_full_submission", "")
    return RAW_DIR / "metadata_json" / ticker / f"{stem}.json"


def read_filing_text(file_path: Path) -> str:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    if file_path.suffix.lower() in {".htm", ".html"}:
        raw = clean_html(raw)
    return remove_boilerplate(raw)


def extract_year(file_path: Path, metadata: Dict) -> int:
    for key in ["filing_year", "report_year", "year"]:
        value = metadata.get(key)
        if value:
            return int(str(value)[:4])

    match = re.search(r"(20\d{2})", file_path.name)
    if not match:
        raise ValueError(f"Cannot infer year from {file_path}")
    return int(match.group(1))


def process_file(file_path: Path, ticker: str) -> List[Dict]:
    metadata = read_metadata(find_metadata_path(file_path))
    text = read_filing_text(file_path)
    sections = extract_sections(text)
    year = extract_year(file_path, metadata)

    records = []
    company_name = metadata.get("company_name")
    filing_date = metadata.get("filing_date")
    accession_number = metadata.get("accession_number")

    for section_name, section_text in sections.items():
        normalized_section = SECTION_MAP.get(section_name.lower(), "10k_full_text")
        doc_id = f"{ticker}_{year}_{normalized_section}"

        records.append({
            "doc_id": doc_id,
            "chunk_id": doc_id,
            "company": ticker,
            "ticker": ticker,
            "year": year,
            "source": "edgar",
            "source_type": "10k_section",
            "doc_type": normalized_section,
            "section": section_name,
            "page_title": f"{ticker} {year} 10-K {normalized_section}",
            "url": None,
            "text": section_text,
            "metadata": {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "company_name": company_name,
                "filing_date": filing_date,
                "accession_number": accession_number,
                "form": metadata.get("form"),
            },
        })

    return records


def iter_filing_files() -> List[Path]:
    txt_dir = RAW_DIR / "txt"
    html_dir = RAW_DIR / "html"

    if txt_dir.exists():
        return sorted(txt_dir.glob("*/*_full_submission.txt"))

    if html_dir.exists():
        return sorted(list(html_dir.glob("*/*.htm")) + list(html_dir.glob("*/*.html")))

    return []


# =========================
# Main Pipeline
# =========================

def run_pipeline():
    all_records = []
    filing_files = iter_filing_files()

    if not filing_files:
        raise FileNotFoundError(f"No EDGAR files found under {RAW_DIR}")

    for file_path in filing_files:
        ticker = file_path.parent.name
        print(f"[INFO] Processing EDGAR filing: {ticker} / {file_path.name}")
        all_records.extend(process_file(file_path, ticker))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved {len(all_records)} EDGAR records to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
