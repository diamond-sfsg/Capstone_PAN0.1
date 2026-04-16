"""Batch scoring for the ten-company purpose-driven dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from main import (
    _aggregate_page_features,
    _build_scoring_templates,
    _extract_page_features,
    _load_records,
    _page_rows,
    _prepare_record,
)
from llm.judge import run_optional_openai_judging
from preprocess.deduplicator import deduplicate_records
from utils.io_utils import ensure_directory, write_csv, write_json

COMPANY_MAP = [
    {
        "company_name": "Amazon",
        "web_dir": "Amazon",
        "linkedin_dir": "Amazon (1)",
        "ticker": "AMZN",
    },
    {
        "company_name": "Apple",
        "web_dir": "Apple_Inc",
        "linkedin_dir": "Apple",
        "ticker": "AAPL",
    },
    {
        "company_name": "Chevron",
        "web_dir": "Chevron_Corporation",
        "linkedin_dir": "Chevron",
        "ticker": "CVX",
    },
    {
        "company_name": "Cisco",
        "web_dir": "Cisco",
        "linkedin_dir": "Cisco (1)",
        "ticker": "CSCO",
    },
    {
        "company_name": "Johnson & Johnson",
        "web_dir": "Johnson_&_Johnson",
        "linkedin_dir": "Johnson_&_Johnson (1)",
        "ticker": "JNJ",
    },
    {
        "company_name": "Meta",
        "web_dir": "Meta_Platforms",
        "linkedin_dir": "Meta",
        "ticker": "META",
    },
    {
        "company_name": "Netflix",
        "web_dir": "Netflix",
        "linkedin_dir": "Netflix (1)",
        "ticker": "NFLX",
    },
    {
        "company_name": "Nvidia",
        "web_dir": "Nvidia",
        "linkedin_dir": "Nvidia (1)",
        "ticker": "NVDA",
    },
    {
        "company_name": "Oracle",
        "web_dir": "Oracle_Corporation",
        "linkedin_dir": "Oracle",
        "ticker": "ORCL",
    },
    {
        "company_name": "Walmart",
        "web_dir": "Walmart",
        "linkedin_dir": "Walmart (1)",
        "ticker": "WMT",
    },
]


def main():
    """Run batch scoring for all configured companies."""
    args = _build_parser().parse_args()
    base_dir = Path(args.data_root)
    output_dir = ensure_directory(args.output_dir)
    company_rows = []

    for company in COMPANY_MAP:
        summary = _score_company(base_dir, output_dir, company, enable_openai_judge=args.enable_openai_judge)
        company_rows.append(summary)
        print(
            f"{company['company_name']}: {summary['aggregate_score']} "
            f"({summary['aggregate_label']}) from {summary['record_count']} records"
        )

    company_rows.sort(key=lambda item: item["aggregate_score"], reverse=True)
    write_csv(output_dir / "all_companies_summary.csv", company_rows)
    write_json(output_dir / "all_companies_summary.json", company_rows)


def _build_parser():
    parser = argparse.ArgumentParser(description="Batch score all ten companies.")
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data" / "data"),
        help="Root directory containing selected_10_companies, linkedin_selected_10_companies, and 10Ks.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "outputs" / "batch"),
        help="Output directory for batch scoring results.",
    )
    parser.add_argument(
        "--enable-openai-judge",
        action="store_true",
        help="Enable optional OpenAI embedding + LLM review when OPENAI_API_KEY is set.",
    )
    return parser


def _score_company(base_dir: Path, output_dir: Path, company: dict, enable_openai_judge=False):
    paths = []
    web_dir = base_dir / "selected_10_companies" / company["web_dir"]
    linkedin_dir = base_dir / "linkedin_selected_10_companies" / company["linkedin_dir"]
    sec_dir = base_dir / "10Ks" / "edgar" / company["ticker"]

    for candidate in [
        web_dir / "pages.csv",
        web_dir / "pages.jsonl",
        linkedin_dir / "pages (1).jsonl",
    ]:
        if candidate.exists():
            paths.append(candidate)

    if sec_dir.exists():
        paths.extend(sorted(sec_dir.glob("*_full_submission.txt")))

    raw_records = []
    for path in paths:
        raw_records.extend(_load_records(path))

    prepared_records = [_prepare_record(record) for record in raw_records]
    prepared_records = [record for record in prepared_records if record.get("text")]
    prepared_records = deduplicate_records(prepared_records)
    page_features = [_extract_page_features(record) for record in prepared_records]
    page_features = _filter_company_evidence(page_features)
    company_summary = _aggregate_page_features(page_features, company["company_name"])

    from scoring.aggregate_score import apply_llm_adjustment, compute_aggregate_score
    from scoring.articulation_score import compute_articulation_score
    from scoring.embedding_score import compute_embedding_score
    from scoring.execution_score import compute_execution_score

    articulation = compute_articulation_score(company_summary)
    embedding = compute_embedding_score(company_summary)
    execution = compute_execution_score(company_summary)
    aggregate = compute_aggregate_score(
        {
            "purpose_articulation": articulation,
            "operational_embedding": embedding,
            "execution_consistency": execution,
        }
    )
    llm_review = {"enabled": False, "status": "disabled"}
    if enable_openai_judge:
        llm_review = run_optional_openai_judging(
            company["company_name"],
            page_features,
            {
                "purpose_articulation": articulation,
                "operational_embedding": embedding,
                "execution_consistency": execution,
                "aggregate": aggregate,
            },
        )
        aggregate = apply_llm_adjustment(aggregate, llm_review)

    stem = _slugify(company["company_name"])
    detail_payload = {
        "company_name": company["company_name"],
        "input_files": [str(path) for path in paths],
        "record_count": len(page_features),
        "scores": {
            "purpose_articulation": articulation,
            "operational_embedding": embedding,
            "execution_consistency": execution,
            "aggregate": aggregate,
        },
        "evidence": {
            "top_pages": company_summary["top_pages"],
            "scoring_top_pages": company_summary.get("scoring_top_pages", []),
            "years_found": company_summary["years_found"],
            "industries_found": company_summary["industries_found"],
            "purpose_examples": company_summary["purpose_examples"],
        },
        "scoring_templates": _build_scoring_templates(
            company_summary,
            {
                "purpose_articulation": articulation,
                "operational_embedding": embedding,
                "execution_consistency": execution,
            },
        ),
        "openai_review": llm_review,
    }
    write_json(output_dir / f"{stem}_summary.json", detail_payload)
    write_csv(output_dir / f"{stem}_pages.csv", _page_rows(page_features))

    return {
        "company_name": company["company_name"],
        "ticker": company["ticker"],
        "record_count": len(page_features),
        "input_file_count": len(paths),
        "purpose_articulation_score": articulation["overall"],
        "operational_embedding_score": embedding["overall"],
        "execution_consistency_score": execution["overall"],
        "aggregate_score": aggregate["overall"],
        "aggregate_label": aggregate["label"],
        "top_evidence_title_1": company_summary["top_pages"][0]["title"] if company_summary["top_pages"] else "",
        "top_evidence_title_2": company_summary["top_pages"][1]["title"] if len(company_summary["top_pages"]) > 1 else "",
        "purpose_example_1": company_summary["purpose_examples"][0] if company_summary["purpose_examples"] else "",
    }


def _filter_company_evidence(page_features):
    filtered = []
    seen_fingerprints = set()
    for page in sorted(page_features, key=lambda item: item["page_signal_score"], reverse=True):
        fingerprint = (
            page.get("title", "").strip().lower(),
            tuple(page.get("purpose_sentences", [])[:2]),
        )
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        filtered.append(page)
    return filtered


def _slugify(value: str):
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


if __name__ == "__main__":
    main()
