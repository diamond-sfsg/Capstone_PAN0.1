from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


VALID_XBRL_PREFIXES = {
    "dei",
    "us-gaap",
    "srt",
    "ecd",
    "country",
    "iso4217",
    "xbrli",
    "xbrldi",
}

NOISE_PREFIXES = {
    "ix",
    "ixt",
    "ixt-sec",
    "link",
    "xlink",
    "xsi",
}

METADATA_FIELDS = [
    "ticker",
    "company_name",
    "cik",
    "form",
    "filing_date",
    "report_date",
    "filing_year",
    "accession_number",
    "primary_document",
    "primary_doc_url",
    "resolved_primary_doc_path",
    "full_submission_txt_url",
    "resolved_full_submission_txt_path",
]

FACT_FIELDS = [
    "ticker",
    "filing_year",
    "filing_date",
    "source_file",
    "category",
    "tag_name",
    "tag_prefix",
    "tag_local_name",
    "context_ref",
    "unit_ref",
    "decimals",
    "scale",
    "sign",
    "value",
]

PURPOSE_TEXT_FIELDS = [
    "ticker",
    "filing_year",
    "filing_date",
    "source_file",
    "text_length",
    "removed_url_count",
    "removed_xbrl_token_count",
    "purpose_text",
]


@dataclass(frozen=True)
class EdgarCleanSummary:
    metadata_files: int
    submissions_files: int
    html_files: int
    valid_metadata_rows: int
    valid_10k_submission_rows: int
    xbrl_fact_rows: int
    invalid_url_rows: int
    noise_rows: int


@dataclass(frozen=True)
class EdgarPurposeTextSummary:
    html_files: int
    output_rows: int
    total_text_length: int
    removed_url_count: int
    removed_xbrl_token_count: int


@dataclass(frozen=True)
class EdgarByTypeSummary:
    metadata_json_files: int
    submissions_json_files: int
    html_files: int
    txt_files: int
    output_files: int
    removed_url_count: int
    removed_xbrl_token_count: int


def clean_edgar_dataset(input_dir: Path, output_dir: Path) -> EdgarCleanSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows, invalid_url_rows = clean_metadata(input_dir)
    submission_rows = clean_submissions(input_dir)
    fact_rows, noise_rows = clean_ixbrl_html(input_dir)

    write_csv(output_dir / "metadata_clean.csv", METADATA_FIELDS, metadata_rows)
    write_csv(output_dir / "submissions_10k_clean.csv", submission_fieldnames(submission_rows), submission_rows)
    write_csv(output_dir / "xbrl_facts_clean.csv", FACT_FIELDS, fact_rows)
    write_csv(
        output_dir / "invalid_urls_and_paths.csv",
        ["source_file", "field", "value", "reason"],
        invalid_url_rows,
    )
    write_csv(
        output_dir / "noise_removed.csv",
        ["source_file", "category", "sample"],
        noise_rows,
    )

    return EdgarCleanSummary(
        metadata_files=len(list(input_dir.glob("*/*_metadata.json"))),
        submissions_files=len(list(input_dir.glob("*/submissions.json"))),
        html_files=len(list(input_dir.glob("*/*.htm"))),
        valid_metadata_rows=len(metadata_rows),
        valid_10k_submission_rows=len(submission_rows),
        xbrl_fact_rows=len(fact_rows),
        invalid_url_rows=len(invalid_url_rows),
        noise_rows=len(noise_rows),
    )


def clean_edgar_purpose_text(input_dir: Path, output_dir: Path) -> EdgarPurposeTextSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for path in sorted(input_dir.glob("*/*.htm")):
        metadata = infer_metadata_from_filename(path)
        raw_html = path.read_text(encoding="utf-8", errors="ignore")
        extracted = extract_visible_purpose_text(raw_html)
        if not extracted["purpose_text"]:
            continue
        rows.append(
            {
                "ticker": metadata["ticker"],
                "filing_year": metadata["filing_year"],
                "filing_date": metadata["filing_date"],
                "source_file": str(path),
                "text_length": str(len(extracted["purpose_text"])),
                "removed_url_count": str(extracted["removed_url_count"]),
                "removed_xbrl_token_count": str(extracted["removed_xbrl_token_count"]),
                "purpose_text": extracted["purpose_text"],
            }
        )

    write_csv(output_dir / "edgar_purpose_text_clean.csv", PURPOSE_TEXT_FIELDS, rows)
    write_jsonl(output_dir / "edgar_purpose_text_clean.jsonl", rows)

    return EdgarPurposeTextSummary(
        html_files=len(list(input_dir.glob("*/*.htm"))),
        output_rows=len(rows),
        total_text_length=sum(int(row["text_length"]) for row in rows),
        removed_url_count=sum(int(row["removed_url_count"]) for row in rows),
        removed_xbrl_token_count=sum(int(row["removed_xbrl_token_count"]) for row in rows),
    )


def clean_edgar_files_by_type(input_dir: Path, output_dir: Path, show_progress: bool = False) -> EdgarByTypeSummary:
    metadata_paths = sorted(input_dir.glob("*/*_metadata.json"))
    submissions_paths = sorted(input_dir.glob("*/submissions.json"))
    html_paths = sorted(input_dir.glob("*/*.htm"))
    txt_paths = sorted(input_dir.glob("*/*_full_submission.txt"))
    total_files = len(metadata_paths) + len(submissions_paths) + len(html_paths) + len(txt_paths)
    processed_files = 0

    counts = {
        "metadata_json": 0,
        "submissions_json": 0,
        "html": 0,
        "txt": 0,
        "output": 0,
        "urls": 0,
        "xbrl_tokens": 0,
    }

    for path in metadata_paths:
        cleaned = clean_metadata_file(path)
        write_json(output_path_for(path, input_dir, output_dir, "metadata_json"), cleaned)
        counts["metadata_json"] += 1
        counts["output"] += 1
        processed_files += 1
        print_progress(processed_files, total_files, "metadata_json", path, input_dir, show_progress)

    for path in submissions_paths:
        cleaned = clean_submissions_file(path)
        write_json(output_path_for(path, input_dir, output_dir, "submissions_json"), cleaned)
        counts["submissions_json"] += 1
        counts["output"] += 1
        processed_files += 1
        print_progress(processed_files, total_files, "submissions_json", path, input_dir, show_progress)

    for path in html_paths:
        raw_html = path.read_text(encoding="utf-8", errors="ignore")
        extracted = extract_visible_purpose_text(raw_html)
        write_text(output_path_for(path, input_dir, output_dir, "html"), str(extracted["purpose_text"]))
        counts["html"] += 1
        counts["output"] += 1
        counts["urls"] += int(extracted["removed_url_count"])
        counts["xbrl_tokens"] += int(extracted["removed_xbrl_token_count"])
        processed_files += 1
        print_progress(processed_files, total_files, "html", path, input_dir, show_progress)

    for path in txt_paths:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        extracted = extract_full_submission_purpose_text(raw_text)
        write_text(output_path_for(path, input_dir, output_dir, "txt"), str(extracted["purpose_text"]))
        counts["txt"] += 1
        counts["output"] += 1
        counts["urls"] += int(extracted["removed_url_count"])
        counts["xbrl_tokens"] += int(extracted["removed_xbrl_token_count"])
        processed_files += 1
        print_progress(processed_files, total_files, "txt", path, input_dir, show_progress)

    return EdgarByTypeSummary(
        metadata_json_files=counts["metadata_json"],
        submissions_json_files=counts["submissions_json"],
        html_files=counts["html"],
        txt_files=counts["txt"],
        output_files=counts["output"],
        removed_url_count=counts["urls"],
        removed_xbrl_token_count=counts["xbrl_tokens"],
    )


def clean_metadata_file(path: Path) -> dict[str, str]:
    data = read_json_object(path)
    return {
        "ticker": clean_scalar(data.get("ticker")),
        "company_name": clean_scalar(data.get("company_name")),
        "cik": normalize_cik(data.get("cik")),
        "form": clean_scalar(data.get("form")),
        "filing_date": clean_scalar(data.get("filing_date")),
        "report_date": clean_scalar(data.get("report_date")),
        "filing_year": clean_scalar(data.get("filing_year")),
        "accession_number": clean_scalar(data.get("accession_number")),
        "accession_number_nodash": clean_scalar(data.get("accession_number_nodash")),
        "primary_document": clean_scalar(data.get("primary_document")),
        "primary_doc_description": clean_scalar(data.get("primary_doc_description")),
        "primary_doc_content_type": clean_scalar(data.get("primary_doc_content_type")),
    }


def clean_submissions_file(path: Path) -> dict[str, Any]:
    data = read_json_object(path)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filings_10k: list[dict[str, str]] = []

    for index, form in enumerate(forms):
        if form != "10-K":
            continue
        filings_10k.append(
            {
                "accession_number": list_value(recent, "accessionNumber", index),
                "filing_date": list_value(recent, "filingDate", index),
                "report_date": list_value(recent, "reportDate", index),
                "form": clean_scalar(form),
                "primary_document": list_value(recent, "primaryDocument", index),
            }
        )

    return {
        "ticker": first_value(data.get("tickers")),
        "company_name": clean_scalar(data.get("name")),
        "cik": normalize_cik(data.get("cik")),
        "entity_type": clean_scalar(data.get("entityType")),
        "sic": clean_scalar(data.get("sic")),
        "sic_description": clean_scalar(data.get("sicDescription")),
        "category": clean_scalar(data.get("category")),
        "fiscal_year_end": clean_scalar(data.get("fiscalYearEnd")),
        "state_of_incorporation": clean_scalar(data.get("stateOfIncorporation")),
        "filings_10k": filings_10k,
    }


def print_progress(
    processed_files: int,
    total_files: int,
    category: str,
    path: Path,
    input_dir: Path,
    show_progress: bool,
) -> None:
    if not show_progress:
        return
    percent = int(processed_files * 100 / total_files) if total_files else 100
    relative_path = path.relative_to(input_dir)
    print(f"[{percent:3d}%] {processed_files}/{total_files} {category}: {relative_path}", flush=True)


def clean_metadata(input_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []

    for path in sorted(input_dir.glob("*/*_metadata.json")):
        data = read_json_object(path)
        source = str(path)
        company_dir = path.parent
        primary_doc = str(data.get("primary_document") or "")
        txt_name = f"{data.get('filing_year')}_10K_{data.get('filing_date')}_full_submission.txt"

        resolved_primary = company_dir / path.name.replace("_metadata.json", ".htm")
        resolved_txt = company_dir / txt_name

        row = {
            "ticker": clean_scalar(data.get("ticker")),
            "company_name": clean_scalar(data.get("company_name")),
            "cik": normalize_cik(data.get("cik")),
            "form": clean_scalar(data.get("form")),
            "filing_date": clean_scalar(data.get("filing_date")),
            "report_date": clean_scalar(data.get("report_date")),
            "filing_year": clean_scalar(data.get("filing_year")),
            "accession_number": clean_scalar(data.get("accession_number")),
            "primary_document": primary_doc,
            "primary_doc_url": clean_sec_url(data.get("primary_doc_url"), source, "primary_doc_url", invalid_rows),
            "resolved_primary_doc_path": str(resolved_primary) if resolved_primary.exists() else "",
            "full_submission_txt_url": clean_sec_url(
                data.get("full_submission_txt_url"),
                source,
                "full_submission_txt_url",
                invalid_rows,
            ),
            "resolved_full_submission_txt_path": str(resolved_txt) if resolved_txt.exists() else "",
        }

        for field_name in ("primary_doc_path", "full_submission_txt_path"):
            original_path = clean_scalar(data.get(field_name))
            if original_path and not Path(original_path).exists():
                invalid_rows.append(
                    {
                        "source_file": source,
                        "field": field_name,
                        "value": original_path,
                        "reason": "local path in metadata does not exist; replaced by resolved path",
                    }
                )

        rows.append(row)

    return rows, invalid_rows


def clean_submissions(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("*/submissions.json")):
        data = read_json_object(path)
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])

        for index, form in enumerate(forms):
            if form != "10-K":
                continue
            rows.append(
                {
                    "ticker": first_value(data.get("tickers")),
                    "company_name": clean_scalar(data.get("name")),
                    "cik": normalize_cik(data.get("cik")),
                    "sic": clean_scalar(data.get("sic")),
                    "sic_description": clean_scalar(data.get("sicDescription")),
                    "fiscal_year_end": clean_scalar(data.get("fiscalYearEnd")),
                    "accession_number": list_value(recent, "accessionNumber", index),
                    "filing_date": list_value(recent, "filingDate", index),
                    "report_date": list_value(recent, "reportDate", index),
                    "form": form,
                    "primary_document": list_value(recent, "primaryDocument", index),
                    "source_file": str(path),
                }
            )
    return rows


def clean_ixbrl_html(input_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    facts: list[dict[str, str]] = []
    noise_rows: list[dict[str, str]] = []

    for path in sorted(input_dir.glob("*/*.htm")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata = infer_metadata_from_filename(path)

        for sample in re.findall(r"<!--(.*?)-->", text[:5000], flags=re.DOTALL):
            noise_rows.append(
                {
                    "source_file": str(path),
                    "category": "noise_comment_or_generator_id",
                    "sample": normalize_whitespace(sample)[:300],
                }
            )

        namespace_samples = re.findall(r'xmlns:[\w-]+="[^"]+"', text[:4000])
        if namespace_samples:
            noise_rows.append(
                {
                    "source_file": str(path),
                    "category": "noise_namespace_declaration",
                    "sample": " ".join(namespace_samples[:8])[:300],
                }
            )

        for match in re.finditer(
            r"<ix:(nonFraction|nonNumeric)\b(?P<attrs>[^>]*)>(?P<value>.*?)</ix:\1>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            attrs = parse_attrs(match.group("attrs"))
            raw_value = match.group("value")
            tag_name = clean_scalar(attrs.get("name"))
            prefix, local_name = split_tag_name(tag_name)
            value = clean_fact_value(raw_value)
            category = classify_fact(tag_name, value)

            if not value and attrs.get("xsi:nil") == "true":
                category = "valid_xbrl_nil"

            facts.append(
                {
                    "ticker": metadata["ticker"],
                    "filing_year": metadata["filing_year"],
                    "filing_date": metadata["filing_date"],
                    "source_file": str(path),
                    "category": category,
                    "tag_name": tag_name,
                    "tag_prefix": prefix,
                    "tag_local_name": local_name,
                    "context_ref": clean_scalar(attrs.get("contextRef") or attrs.get("contextref")),
                    "unit_ref": clean_scalar(attrs.get("unitRef") or attrs.get("unitref")),
                    "decimals": clean_scalar(attrs.get("decimals")),
                    "scale": clean_scalar(attrs.get("scale")),
                    "sign": clean_scalar(attrs.get("sign")),
                    "value": value,
                }
            )

    return facts, noise_rows


def classify_fact(tag_name: str, value: str) -> str:
    prefix, _ = split_tag_name(tag_name)
    if prefix in VALID_XBRL_PREFIXES or prefix not in NOISE_PREFIXES:
        if is_taxonomy_reference(value):
            return "valid_xbrl_taxonomy_reference"
        return "valid_xbrl_fact"
    return "noise_xbrl_internal_tag"


def clean_sec_url(value: Any, source: str, field: str, invalid_rows: list[dict[str, str]]) -> str:
    url = clean_scalar(value)
    if not url:
        return ""

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "www.sec.gov" or "/Archives/edgar/" not in parsed.path:
        invalid_rows.append(
            {
                "source_file": source,
                "field": field,
                "value": url,
                "reason": "not a canonical SEC archive URL",
            }
        )
        return ""
    return url


def is_taxonomy_reference(value: str) -> bool:
    if not value:
        return False
    parts = value.split()
    return all(part.startswith(("http://fasb.org/", "http://xbrl.sec.gov/", "http://www.xbrl.org/")) and "#" in part for part in parts)


def clean_fact_value(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return normalize_whitespace(value)


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return normalize_whitespace(str(value))


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_cik(value: Any) -> str:
    cik = re.sub(r"\D", "", clean_scalar(value))
    return cik.zfill(10) if cik else ""


def parse_attrs(value: str) -> dict[str, str]:
    return {
        key: html.unescape(raw_value)
        for key, raw_value in re.findall(r'([\w:-]+)\s*=\s*"([^"]*)"', value)
    }


def split_tag_name(value: str) -> tuple[str, str]:
    if ":" not in value:
        return "", value
    prefix, local_name = value.split(":", 1)
    return prefix, local_name


def infer_metadata_from_filename(path: Path) -> dict[str, str]:
    match = re.match(r"(?P<year>\d{4})_10K_(?P<date>\d{4}-\d{2}-\d{2})", path.name)
    return {
        "ticker": path.parent.name,
        "filing_year": match.group("year") if match else "",
        "filing_date": match.group("date") if match else "",
    }


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        return {}
    return data


def first_value(value: Any) -> str:
    if isinstance(value, list) and value:
        return clean_scalar(value[0])
    return clean_scalar(value)


def list_value(data: dict[str, Any], key: str, index: int) -> str:
    values = data.get(key, [])
    if isinstance(values, list) and index < len(values):
        return clean_scalar(values[index])
    return ""


def submission_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return [
            "ticker",
            "company_name",
            "cik",
            "sic",
            "sic_description",
            "fiscal_year_end",
            "accession_number",
            "filing_date",
            "report_date",
            "form",
            "primary_document",
            "source_file",
        ]
    return list(rows[0].keys())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def output_path_for(path: Path, input_dir: Path, output_dir: Path, category: str) -> Path:
    relative = path.relative_to(input_dir)
    return output_dir / category / relative


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in {
            "script",
            "style",
            "noscript",
            "head",
            "title",
            "ix:hidden",
            "ix:header",
            "ix:references",
            "ix:resources",
            "link:schemaref",
            "xbrli:context",
            "xbrli:unit",
        }:
            self.skip_stack.append(tag_lower)
            return
        if tag_lower in {"br", "p", "div", "tr", "li", "table", "hr"} and not self.skip_stack:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self.skip_stack and self.skip_stack[-1] == tag_lower:
            self.skip_stack.pop()
            return
        if tag_lower in {"p", "div", "tr", "li", "table"} and not self.skip_stack:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_stack:
            return
        if data and data.strip():
            self.parts.append(data)

    def handle_comment(self, data: str) -> None:
        return

    def get_text(self) -> str:
        return "\n".join(self.parts)


def extract_visible_purpose_text(raw_html: str) -> dict[str, str | int]:
    parser = VisibleTextParser()
    parser.feed(raw_html)
    text = html.unescape(parser.get_text())

    text, removed_url_count = remove_urls(text)
    text, removed_xbrl_token_count = remove_xbrl_like_tokens(text)
    text = remove_boilerplate_symbols(text)
    text = normalize_text_lines(text)

    return {
        "purpose_text": text,
        "removed_url_count": removed_url_count,
        "removed_xbrl_token_count": removed_xbrl_token_count,
    }


def extract_full_submission_purpose_text(raw_text: str) -> dict[str, str | int]:
    document_text = extract_10k_document_body(raw_text)
    if re.search(r"<html|<body|<ix:", document_text, flags=re.IGNORECASE):
        return extract_visible_purpose_text(document_text)

    text = html.unescape(document_text)
    text, removed_url_count = remove_urls(text)
    text, removed_xbrl_token_count = remove_xbrl_like_tokens(text)
    text = remove_sec_submission_headers(text)
    text = remove_boilerplate_symbols(text)
    text = normalize_text_lines(text)

    return {
        "purpose_text": text,
        "removed_url_count": removed_url_count,
        "removed_xbrl_token_count": removed_xbrl_token_count,
    }


def extract_10k_document_body(raw_text: str) -> str:
    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw_text, flags=re.IGNORECASE | re.DOTALL)
    for document in documents:
        type_match = re.search(r"<TYPE>\s*([^\r\n<]+)", document, flags=re.IGNORECASE)
        if type_match and type_match.group(1).strip().upper() == "10-K":
            text_match = re.search(r"<TEXT>(.*?)</TEXT>", document, flags=re.IGNORECASE | re.DOTALL)
            return text_match.group(1) if text_match else document
    return raw_text


def remove_sec_submission_headers(text: str) -> str:
    patterns = [
        r"<SEC-DOCUMENT>.*?(?=<DOCUMENT>|$)",
        r"<SEC-HEADER>.*?</SEC-HEADER>",
        r"<ACCEPTANCE-DATETIME>[^\n\r]*",
        r"<(?:DOCUMENT|TYPE|SEQUENCE|FILENAME|DESCRIPTION|TEXT)>[^\n\r]*",
        r"</(?:DOCUMENT|TEXT)>",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def remove_urls(text: str) -> tuple[str, int]:
    patterns = [
        r"https?://\S+",
        r"www\.\S+",
        r"\b[a-zA-Z0-9.-]+\.(?:com|org|net|gov|edu|io|co)\S*",
    ]
    count = 0
    for pattern in patterns:
        text, removed = re.subn(pattern, " ", text)
        count += removed
    return text, count


def remove_xbrl_like_tokens(text: str) -> tuple[str, int]:
    patterns = [
        r"\b(?:dei|us-gaap|srt|ecd|country|iso4217|xbrli|xbrldi|ix|ixt|ixt-sec|xlink|xsi|link):[A-Za-z0-9_.:-]+\b",
        r"\b[a-z]{2,8}:[A-Za-z][A-Za-z0-9_.:-]+\b",
        r"\b(?:contextRef|unitRef|continuedAt|decimals|scale|format|xmlns|schemaRef|xsi:nil)\b",
        r"\b[crdfgi]-[0-9a-f]{1,8}\b",
        r"\b[0-9a-f]{12,}\b",
    ]
    count = 0
    for pattern in patterns:
        text, removed = re.subn(pattern, " ", text, flags=re.IGNORECASE)
        count += removed
    return text, count


def remove_boilerplate_symbols(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[{}<>\\]+", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\bXML\b|\bXHTML\b|\bXBRL\b|\bInline XBRL\b", " ", text, flags=re.IGNORECASE)
    return text


def normalize_text_lines(text: str) -> str:
    raw_lines = merge_wrapped_subject_lines([normalize_whitespace(line) for line in text.splitlines()])
    useful_lines: list[str] = []
    seen: set[str] = set()

    for line in raw_lines:
        if not is_useful_purpose_line(line):
            continue
        compact_key = line.lower()
        if compact_key in seen:
            continue
        seen.add(compact_key)
        useful_lines.append(line)

    return "\n".join(useful_lines)


def merge_wrapped_subject_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            line
            and len(line) < 25
            and next_line
            and re.match(r"^[®™\s]*(is|are|includes|based|operating|as well as|,)", next_line, flags=re.IGNORECASE)
        ):
            merged.append(f"{line} {next_line}")
            index += 2
            continue
        merged.append(line)
        index += 1
    return merged


def is_useful_purpose_line(line: str) -> bool:
    if len(line) < 25:
        return False
    if is_sec_cover_boilerplate(line):
        return False
    if not re.search(r"[A-Za-z]", line):
        return False
    alpha_ratio = len(re.findall(r"[A-Za-z]", line)) / max(len(line), 1)
    if alpha_ratio < 0.35:
        return False
    if re.search(r"<[^>]+>|xmlns|http://|https://|www\.", line, flags=re.IGNORECASE):
        return False
    if re.search(r"\b(?:us-gaap|dei|xbrli|xbrldi|contextRef|unitRef|schemaRef)\b", line, flags=re.IGNORECASE):
        return False
    return True


def is_sec_cover_boilerplate(line: str) -> bool:
    patterns = [
        r"securities and exchange commission",
        r"annual report pursuant to section",
        r"transition report pursuant to section",
        r"for the fiscal year ended",
        r"for the transition period from",
        r"exact name of registrant",
        r"state or other jurisdiction of incorporation",
        r"state or other jurisdiction",
        r"of incorporation or organization",
        r"i\.r\.s\. employer identification no",
        r"address of principal executive offices",
        r"registrant.?s telephone number",
        r"securities registered pursuant to section",
        r"name of exchange on which registered",
        r"name of each exchange on which registered",
        r"common stock.*par value",
        r"nasdaq stock market",
        r"new york stock exchange",
        r"do not check if a smaller reporting company",
        r"indicate by check mark",
        r"large accelerated filer",
        r"accelerated filer",
        r"non-accelerated filer",
        r"smaller reporting company",
        r"emerging growth company",
        r"aggregate market value of the voting and non-voting stock",
        r"shares of common stock were issued and outstanding",
        r"shares of common stock held by executive officers",
        r"documents incorporated by reference",
        r"definitive proxy statement",
        r"unresolved staff comments",
        r"market for registrant.?s common equity",
        r"financial statements and supplementary data",
        r"changes in and disagreements with accountants",
        r"directors, executive officers and corporate governance",
        r"security ownership of certain beneficial owners",
        r"certain relationships and related transactions",
        r"principal accountant fees and services",
        r"exhibit and financial statement schedules",
        r"this annual report on form 10-k.*forward-looking statements",
        r"forward-looking statements are not guarantees",
        r"the company assumes no obligation to revise or update",
        r"unless otherwise stated, all information presented herein",
        r"^table of contents$",
        r"^page\s+\d+$",
    ]
    lowered = line.lower().strip()
    return any(re.search(pattern, lowered) for pattern in patterns)
