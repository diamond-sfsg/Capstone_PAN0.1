from __future__ import annotations

import json
import re
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


# =========================
# Basic Config
# =========================

REPO_ROOT = Path(__file__).resolve().parents[2]

# One ticker per line. Blank lines and # comments are ignored.
TICKER_FILE = REPO_ROOT / "configs" / "company_ticker.txt"

# How many most recent 10-K filings to download per company
MAX_FILINGS_PER_TICKER = 10

# Root output folder. Keep this aligned with the existing raw download folder.
RAW_ROOT = Path(r"D:\BaiduNetdiskDownload\edgar_10k_raw\edgar_10k_raw")

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


def read_tickers(path: Path) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        ticker = raw_line.strip().upper()
        if not ticker or ticker.startswith("#"):
            continue
        if ticker in seen:
            continue
        tickers.append(ticker)
        seen.add(ticker)
    return tickers


def existing_nonempty_file(path: Path | str | None) -> bool:
    if not path:
        return False
    candidate = Path(path)
    return candidate.is_file() and candidate.stat().st_size > 0


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

    if existing_nonempty_file(primary_doc_path):
        print(f"  [SKIP] primary document exists: {primary_doc_path.name}")
        return {
            "primary_doc_url": primary_doc_url,
            "primary_doc_path": str(primary_doc_path),
            "primary_doc_content_type": None,
        }

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

    if existing_nonempty_file(txt_path):
        print(f"  [SKIP] full submission txt exists: {txt_path.name}")
        return {
            "full_submission_txt_url": txt_url,
            "full_submission_txt_path": str(txt_path),
        }

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


def metadata_path_for_filing(filing: FilingRecord, ticker_dir: Path) -> Path:
    return ticker_dir / f"{filing.filing_year}_10K_{filing.filing_date}_metadata.json"


def expected_primary_doc_path(filing: FilingRecord, ticker_dir: Path) -> Path:
    ext = Path(filing.primary_document).suffix.lower() or ".html"
    return ticker_dir / f"{filing.filing_year}_10K_{filing.filing_date}{ext}"


def expected_full_submission_txt_path(filing: FilingRecord, ticker_dir: Path) -> Path:
    return ticker_dir / f"{filing.filing_year}_10K_{filing.filing_date}_full_submission.txt"


def filing_is_complete(filing: FilingRecord, ticker_dir: Path) -> bool:
    manifest_path = metadata_path_for_filing(filing, ticker_dir)
    primary_path = expected_primary_doc_path(filing, ticker_dir)
    txt_path = expected_full_submission_txt_path(filing, ticker_dir)

    if not (
        existing_nonempty_file(manifest_path)
        and existing_nonempty_file(primary_path)
        and existing_nonempty_file(txt_path)
    ):
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return (
        manifest.get("accession_number") == filing.accession_number
        and manifest.get("primary_document") == filing.primary_document
    )


def summarize_ticker_dir(ticker: str, output_root: Path) -> dict[str, Any]:
    ticker_dir = output_root / ticker
    if not ticker_dir.exists():
        return {
            "ticker": ticker,
            "ticker_dir_exists": False,
            "metadata_files": 0,
            "primary_documents": 0,
            "full_submission_txt": 0,
        }

    return {
        "ticker": ticker,
        "ticker_dir_exists": True,
        "metadata_files": len(list(ticker_dir.glob("*_metadata.json"))),
        "primary_documents": len(
            [
                path
                for path in ticker_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".htm", ".html", ".txt"}
                and not path.name.endswith("_full_submission.txt")
            ]
        ),
        "full_submission_txt": len(list(ticker_dir.glob("*_full_submission.txt"))),
    }


# =========================
# Main per ticker
# =========================

def process_ticker(
    ticker: str,
    mapping: dict[str, dict[str, str]],
    output_root: Path,
    check_only: bool = False,
) -> dict[str, Any]:
    ticker = ticker.upper()

    status: dict[str, Any] = {
        **summarize_ticker_dir(ticker, output_root),
        "status": "unknown",
        "filings_found": 0,
        "filings_complete": 0,
        "filings_downloaded_or_repaired": 0,
        "filing_errors": 0,
        "error": None,
    }

    if ticker not in mapping:
        print(f"[ERROR] {ticker}: not found in SEC ticker mapping")
        status["status"] = "not_found_in_sec_mapping"
        return status

    cik = mapping[ticker]["cik_str"]
    company_name = mapping[ticker]["title"]

    print(f"\n=== Processing {ticker} ===")
    print(f"CIK: {cik} | Company: {company_name}")

    ticker_dir = output_root / ticker
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
        status["status"] = "no_10k_found"
        return status

    print(f"Found {len(filings)} recent 10-K filings for {ticker}")
    status["filings_found"] = len(filings)

    for filing in filings:
        if filing_is_complete(filing, ticker_dir):
            status["filings_complete"] += 1
            print(f"  [OK] complete: {filing.filing_date} | {filing.accession_number}")
            continue

        if check_only:
            print(f"  [MISSING] incomplete: {filing.filing_date} | {filing.accession_number}")
            continue

        try:
            print(f"\n- repair/download {filing.filing_date} | {filing.accession_number}")
            download_info = download_primary_document(filing, ticker_dir)
            txt_info = try_download_full_submission_txt(filing, ticker_dir)
            save_filing_manifest(filing, ticker_dir, download_info, txt_info)
            status["filings_downloaded_or_repaired"] += 1
            print("  [OK] saved")
        except Exception as e:
            status["filing_errors"] += 1
            print(f"  [ERROR] failed filing {filing.accession_number}: {e}")

    if check_only:
        incomplete = status["filings_found"] - status["filings_complete"]
        status["status"] = "complete" if incomplete == 0 else "incomplete"
    else:
        status["status"] = "ok" if status["filing_errors"] == 0 else "partial_error"
    return status


def write_progress(output_root: Path, results: list[dict[str, Any]]) -> None:
    write_json(output_root / "_download_progress.json", results)


def write_summary_csv(output_root: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return

    columns = list(dict.fromkeys(key for row in results for key in row))
    lines = [",".join(columns)]
    for row in results:
        values = []
        for column in columns:
            value = "" if row.get(column) is None else str(row.get(column))
            values.append('"' + value.replace('"', '""') + '"')
        lines.append(",".join(values))

    write_text(output_root / "_download_summary.csv", "\n".join(lines) + "\n")


# =========================
# Main
# =========================

def parse_args() -> Any:
    parser = ArgumentParser(
        description="Check and repair EDGAR 10-K downloads for tickers in configs/company_ticker.txt."
    )
    parser.add_argument("--ticker-file", type=Path, default=TICKER_FILE)
    parser.add_argument("--output-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--max-filings", type=int, default=MAX_FILINGS_PER_TICKER)
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based ticker position to start from in the ticker file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report missing/incomplete filings; do not download files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    ensure_dir(output_root)

    try:
        mapping = load_ticker_cik_mapping()
    except Exception as e:
        print(f"[FATAL] failed to load ticker mapping: {e}")
        return

    tickers = read_tickers(args.ticker_file)
    print(f"Loaded {len(tickers)} tickers from {args.ticker_file}")
    print(f"Output root: {output_root}")

    global MAX_FILINGS_PER_TICKER
    MAX_FILINGS_PER_TICKER = args.max_filings

    start_index = max(args.start_index, 1)
    selected_tickers = tickers[start_index - 1 :]

    results: list[dict[str, Any]] = []
    for index, ticker in enumerate(selected_tickers, start=start_index):
        print(f"\n##### {index}/{len(tickers)} #####")
        try:
            result = process_ticker(
                ticker=ticker,
                mapping=mapping,
                output_root=output_root,
                check_only=args.check_only,
            )
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
            result = {
                "ticker": ticker,
                "status": "error",
                "error": str(e),
            }
        results.append(result)
        write_progress(output_root, results)

    write_summary_csv(output_root, results)
    print(f"\nDone. Summary saved to: {output_root / '_download_summary.csv'}")


if __name__ == "__main__":
    main()
