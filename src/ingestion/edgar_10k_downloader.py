from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


# =========================
# Basic Config
# =========================

TICKERS = [
    "AAPL",
    "AMZN",
    "CSCO",
    "CVX",
    "JNJ",
    "META",
    "NFLX",
    "NVDA",
    "ORCL",
    "WMT",
]

# How many most recent 10-K filings to download per company
MAX_FILINGS_PER_TICKER = 10

# Root output folder
RAW_ROOT = Path("data/raw/edgar")

# SEC requires a descriptive User-Agent with contact info
USER_AGENT = "UCI MSBA RAG Project ruofay3l@uci.edu"

# polite request gap
REQUEST_SLEEP_SECONDS = 0.3

# request timeout
TIMEOUT = 30


# =========================
# Helpers
# =========================

def make_headers(host: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }
    if host:
        headers["Host"] = host
    return headers


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
def http_get(url: str, headers: dict[str, str], timeout: int = TIMEOUT) -> requests.Response:
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def cik_10digit(cik: str | int) -> str:
    return str(cik).zfill(10)


def cik_nopad(cik: str | int) -> str:
    return str(int(str(cik)))


def accession_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def maybe_sleep() -> None:
    time.sleep(REQUEST_SLEEP_SECONDS)


@dataclass
class FilingRecord:
    ticker: str
    cik: str
    company_name: str
    accession_number: str
    filing_date: str
    report_date: str | None
    form: str
    primary_document: str
    primary_doc_description: str | None

    @property
    def accession_nodash(self) -> str:
        return accession_no_dashes(self.accession_number)

    @property
    def filing_year(self) -> str:
        return self.filing_date[:4] if self.filing_date else "unknown"


# =========================
# SEC Mapping
# =========================

def load_ticker_cik_mapping() -> dict[str, dict[str, str]]:
    """
    Load SEC ticker->CIK mapping from:
    https://www.sec.gov/files/company_tickers.json
    """
    print("Loading SEC ticker -> CIK mapping...")
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = http_get(url, headers=make_headers("www.sec.gov"))
    maybe_sleep()
    raw = resp.json()

    mapping: dict[str, dict[str, str]] = {}
    for _, item in raw.items():
        ticker = item["ticker"].upper()
        mapping[ticker] = {
            "cik_str": str(item["cik_str"]),
            "title": item["title"],
        }
    return mapping


# =========================
# Submission JSON
# =========================

def load_company_submissions(cik: str) -> dict[str, Any]:
    """
    SEC submissions endpoint:
    https://data.sec.gov/submissions/CIK##########.json
    """
    cik_padded = cik_10digit(cik)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    resp = http_get(url, headers=make_headers("data.sec.gov"))
    maybe_sleep()
    return resp.json()


def extract_recent_10k_filings(
    ticker: str,
    cik: str,
    company_name: str,
    submissions_json: dict[str, Any],
    max_filings: int = MAX_FILINGS_PER_TICKER,
) -> list[FilingRecord]:
    """
    Parse recent filings from submissions JSON and keep only 10-K.
    """
    recent = submissions_json.get("filings", {}).get("recent", {})

    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    forms = recent.get("form", [])
    primary_documents = recent.get("primaryDocument", [])
    primary_doc_descs = recent.get("primaryDocDescription", [])

    filings: list[FilingRecord] = []

    for i, form in enumerate(forms):
        if form != "10-K":
            continue

        filing = FilingRecord(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            accession_number=accession_numbers[i],
            filing_date=filing_dates[i],
            report_date=report_dates[i] if i < len(report_dates) else None,
            form=form,
            primary_document=primary_documents[i],
            primary_doc_description=primary_doc_descs[i] if i < len(primary_doc_descs) else None,
        )
        filings.append(filing)

        if len(filings) >= max_filings:
            break

    return filings


# =========================
# Filing Download
# =========================

def build_archive_base_url(cik: str, accession_number: str) -> str:
    """
    SEC archive path:
    /Archives/edgar/data/{cik_nopad}/{accession_no_dashes}/
    """
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_nopad(cik)}/{accession_no_dashes(accession_number)}"
    )


def download_primary_document(filing: FilingRecord, output_dir: Path) -> dict[str, str | None]:
    """
    Download the filing's primary document.
    Usually HTML/HTM; sometimes TXT.
    """
    archive_base = build_archive_base_url(filing.cik, filing.accession_number)
    primary_doc_url = f"{archive_base}/{filing.primary_document}"

    ext = Path(filing.primary_document).suffix.lower()
    if not ext:
        ext = ".html"

    filename_base = f"{filing.filing_year}_10K_{filing.filing_date}"
    primary_doc_path = output_dir / f"{filename_base}{ext}"

    print(f"  Downloading primary document: {primary_doc_url}")
    resp = http_get(primary_doc_url, headers=make_headers("www.sec.gov"))
    maybe_sleep()

    content_type = resp.headers.get("Content-Type", "")
    write_text(primary_doc_path, resp.text)

    return {
        "primary_doc_url": primary_doc_url,
        "primary_doc_path": str(primary_doc_path),
        "primary_doc_content_type": content_type,
    }


def try_download_full_submission_txt(filing: FilingRecord, output_dir: Path) -> dict[str, str | None]:
    """
    Try downloading the full submission TXT:
    https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{accession_number}.txt
    """
    archive_base = build_archive_base_url(filing.cik, filing.accession_number)
    txt_url = f"{archive_base}/{filing.accession_number}.txt"

    filename_base = f"{filing.filing_year}_10K_{filing.filing_date}_full_submission"
    txt_path = output_dir / f"{filename_base}.txt"

    try:
        print(f"  Trying full submission txt: {txt_url}")
        resp = http_get(txt_url, headers=make_headers("www.sec.gov"))
        maybe_sleep()
        write_text(txt_path, resp.text)

        return {
            "full_submission_txt_url": txt_url,
            "full_submission_txt_path": str(txt_path),
        }
    except Exception as e:
        print(f"  [WARN] full submission txt not downloaded: {e}")
        return {
            "full_submission_txt_url": None,
            "full_submission_txt_path": None,
        }


def save_filing_manifest(
    filing: FilingRecord,
    ticker_dir: Path,
    download_info: dict[str, str | None],
    txt_info: dict[str, str | None],
) -> None:
    manifest = {
        "ticker": filing.ticker,
        "company_name": filing.company_name,
        "cik": filing.cik,
        "form": filing.form,
        "filing_date": filing.filing_date,
        "report_date": filing.report_date,
        "filing_year": filing.filing_year,
        "accession_number": filing.accession_number,
        "accession_number_nodash": filing.accession_nodash,
        "primary_document": filing.primary_document,
        "primary_doc_description": filing.primary_doc_description,
        **download_info,
        **txt_info,
    }

    manifest_name = f"{filing.filing_year}_10K_{filing.filing_date}_metadata.json"
    manifest_path = ticker_dir / manifest_name
    write_json(manifest_path, manifest)


# =========================
# Main per ticker
# =========================

def process_ticker(ticker: str, mapping: dict[str, dict[str, str]]) -> None:
    ticker = ticker.upper()

    if ticker not in mapping:
        print(f"[ERROR] {ticker}: not found in SEC ticker mapping")
        return

    cik = mapping[ticker]["cik_str"]
    company_name = mapping[ticker]["title"]

    print(f"\n=== Processing {ticker} ===")
    print(f"CIK: {cik} | Company: {company_name}")

    ticker_dir = RAW_ROOT / ticker
    ensure_dir(ticker_dir)

    submissions = load_company_submissions(cik)

    # save raw submissions json
    submissions_path = ticker_dir / "submissions.json"
    write_json(submissions_path, submissions)

    filings = extract_recent_10k_filings(
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        submissions_json=submissions,
        max_filings=MAX_FILINGS_PER_TICKER,
    )

    if not filings:
        print(f"[WARN] {ticker}: no 10-K filings found in recent submissions")
        return

    print(f"Found {len(filings)} recent 10-K filings for {ticker}")

    for filing in filings:
        try:
            print(f"\n- {filing.filing_date} | {filing.accession_number}")
            download_info = download_primary_document(filing, ticker_dir)
            txt_info = try_download_full_submission_txt(filing, ticker_dir)
            save_filing_manifest(filing, ticker_dir, download_info, txt_info)
            print("  [OK] saved")
        except Exception as e:
            print(f"  [ERROR] failed filing {filing.accession_number}: {e}")


# =========================
# Main
# =========================

def main() -> None:
    ensure_dir(RAW_ROOT)

    try:
        mapping = load_ticker_cik_mapping()
    except Exception as e:
        print(f"[FATAL] failed to load ticker mapping: {e}")
        return

    for ticker in TICKERS:
        try:
            process_ticker(ticker, mapping)
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")


if __name__ == "__main__":
    main()