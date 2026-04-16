"""Application entry point for purpose-driven scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from features.employee_features import extract_employee_features
from features.execution_features import extract_execution_features
from features.leadership_features import extract_leadership_features
from features.purpose_features import extract_purpose_features
from features.strategy_features import extract_strategy_features
from ingest.load_csv import load_csv
from ingest.load_json import load_json
from ingest.load_txt import load_txt
from preprocess.cleaner import clean_text
from preprocess.deduplicator import deduplicate_records
from preprocess.splitter import split_text
from config.scale_definitions import DIMENSION_DEFINITIONS
from config.scoring_config import (
    DEFAULT_TOP_EVIDENCE_COUNT,
    DEFAULT_CHUNK_WORD_SIZE,
    DISPLAY_EXCLUDE_TITLE_PATTERNS,
    DISPLAY_EXCLUDE_URL_PATTERNS,
    GENERIC_EVIDENCE_TITLE_PATTERNS,
    PURPOSE_KEYWORDS,
)
from llm.judge import run_optional_openai_judging
from scoring.aggregate_score import apply_llm_adjustment, compute_aggregate_score
from scoring.articulation_score import compute_articulation_score
from scoring.embedding_score import compute_embedding_score
from scoring.execution_score import compute_execution_score
from utils.io_utils import ensure_directory, write_csv, write_json
from utils.text_utils import keyword_density_score, safe_ratio


def main():
    """Run the purpose-driven scoring pipeline."""
    args = _build_parser().parse_args()
    input_paths = _resolve_input_paths(args)
    if not input_paths:
        raise SystemExit("No input files found. Pass --input files or --input-dir.")

    raw_records = []
    for path in input_paths:
        raw_records.extend(_load_records(path))

    prepared_records = [_prepare_record(record) for record in raw_records]
    prepared_records = [record for record in prepared_records if record["text"]]
    prepared_records = deduplicate_records(prepared_records)
    page_features = [_extract_page_features(record) for record in prepared_records]
    company_summary = _aggregate_page_features(page_features, args.company_name)

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
    if args.enable_openai_judge:
        llm_review = run_optional_openai_judging(
            args.company_name,
            page_features,
            {
                "purpose_articulation": articulation,
                "operational_embedding": embedding,
                "execution_consistency": execution,
                "aggregate": aggregate,
            },
        )
        aggregate = apply_llm_adjustment(aggregate, llm_review)

    output_dir = ensure_directory(args.output_dir)
    summary_payload = {
        "company_name": company_summary["company_name"],
        "input_files": [str(path) for path in input_paths],
        "record_count": len(prepared_records),
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
        "scale_definition": DIMENSION_DEFINITIONS,
        "scoring_templates": _build_scoring_templates(
            company_summary,
            {
                "purpose_articulation": articulation,
                "operational_embedding": embedding,
                "execution_consistency": execution,
            },
        ),
        "openai_review": llm_review,
        "notes": [
            "This is a heuristic text-analysis implementation of the provided scale.",
            "Historical consistency is approximated from year and narrative coverage in the available documents.",
            "Scores are strongest when the input includes annual reports, leadership letters, sustainability reports, careers pages, and strategy disclosures.",
        ],
    }

    json_path = output_dir / f"{args.output_stem}_summary.json"
    csv_path = output_dir / f"{args.output_stem}_pages.csv"
    write_json(json_path, summary_payload)
    write_csv(csv_path, _page_rows(page_features))

    print(f"Company: {company_summary['company_name']}")
    print(f"Records analyzed: {len(prepared_records)}")
    print(f"Aggregate score: {aggregate['overall']} ({aggregate['label']})")
    print(f"Purpose articulation: {articulation['overall']}")
    print(f"Operational embedding: {embedding['overall']}")
    print(f"Execution consistency: {execution['overall']}")
    print(f"Summary JSON: {json_path}")
    print(f"Page CSV: {csv_path}")


def _build_scoring_templates(company_summary, group_scores):
    templates = {}
    top_titles = [page["title"] for page in company_summary.get("top_pages", [])]
    for group_name, group_meta in DIMENSION_DEFINITIONS.items():
        group_result = group_scores[group_name]
        templates[group_name] = {
            "title": group_meta["title"],
            "overall_score": group_result["overall"],
            "dimensions": {},
        }
        for dimension_name, dimension_meta in group_meta["dimensions"].items():
            score = int(group_result["dimensions"].get(dimension_name, 0))
            templates[group_name]["dimensions"][dimension_name] = {
                "question": dimension_meta["question"],
                "score": score,
                "score_description": dimension_meta["score_descriptions"][str(score)],
                "evidence_template": {
                    "required_signals": dimension_meta["evidence_template"]["required_signals"],
                    "preferred_sources": dimension_meta["evidence_template"]["preferred_sources"],
                    "suggested_evidence_titles": top_titles[:3],
                    "suggested_quote_candidates": company_summary.get("purpose_examples", [])[:2],
                },
            }
    return templates


def _build_parser():
    parser = argparse.ArgumentParser(description="Score whether a company appears purpose-driven.")
    parser.add_argument("--input", nargs="*", default=[], help="Input files (.csv, .json, .jsonl, .txt).")
    parser.add_argument("--input-dir", default="", help="Optional directory to scan for input files.")
    parser.add_argument("--company-name", default="Unknown Company", help="Company name for reporting.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "outputs"),
        help="Directory for summary outputs.",
    )
    parser.add_argument("--output-stem", default="purpose_driven", help="Output filename stem.")
    parser.add_argument(
        "--enable-openai-judge",
        action="store_true",
        help="Enable optional OpenAI embedding + LLM judgment if OPENAI_API_KEY is set.",
    )
    return parser


def _resolve_input_paths(args):
    paths = [Path(path).expanduser() for path in args.input]
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser()
        if input_dir.exists():
            for suffix in ("*.csv", "*.json", "*.jsonl", "*.txt"):
                paths.extend(sorted(input_dir.glob(suffix)))
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


def _load_records(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix in {".json", ".jsonl"}:
        return load_json(path)
    if suffix == ".txt":
        return load_txt(path)
    return []


def _prepare_record(record):
    text = clean_text(record.get("text", ""))
    chunks = split_text(text, max_words=DEFAULT_CHUNK_WORD_SIZE)
    merged_text = _select_relevant_chunks(chunks)
    prepared = dict(record)
    prepared["text"] = merged_text
    prepared["chunk_count"] = len(chunks)
    return prepared


def _select_relevant_chunks(chunks):
    if not chunks:
        return ""
    keywords = []
    for values in PURPOSE_KEYWORDS.values():
        keywords.extend(values)
    scored = []
    for index, chunk in enumerate(chunks):
        score = keyword_density_score(chunk, keywords)
        if score > 0:
            scored.append((score, index, chunk))
    if not scored:
        return " ".join(chunks[:12])
    top_scored = sorted(scored, key=lambda item: (-item[0], item[1]))[:12]
    top_scored = sorted(top_scored, key=lambda item: item[1])
    return " ".join(chunk for _, _, chunk in top_scored)


def _extract_page_features(record):
    source_bucket = _classify_source(record.get("source_file", ""))
    features = {
        "record_id": record.get("record_id"),
        "title": record.get("title") or "(untitled)",
        "url": record.get("url", ""),
        "source_file": record.get("source_file", ""),
        "source_bucket": source_bucket,
        "is_probably_relevant": bool(record.get("is_probably_relevant", False)),
        "url_importance_score": float(record.get("url_importance_score", 0) or 0),
        "text_length": len(record.get("text", "")),
    }
    features.update(extract_purpose_features(record))
    features.update(extract_leadership_features(record))
    features.update(extract_strategy_features(record))
    features.update(extract_employee_features(record))
    features.update(extract_execution_features(record))
    features["page_signal_score"] = _page_signal_score(features)
    features["display_evidence_score"] = _display_evidence_score(features)
    return features


def _page_signal_score(features):
    score = 0.0
    score += min(features["purpose_hits"], 5) * 1.5
    score += min(features["impact_hits"], 8) * 0.8
    score += min(features["strategy_hits"], 6) * 0.7
    score += min(features["employee_hits"], 6) * 0.5
    score += min(features["execution_hits"], 6) * 0.5
    score += min(features["measurement_hits"], 6) * 0.6
    score += min(features["leadership_purpose_co_mentions"], 4) * 1.0
    score += features["url_importance_score"] * 0.4
    if features["is_probably_relevant"]:
        score += 2
    if features.get("purpose_sentence_count", 0) > 0:
        score += 2.5
    if features.get("stakeholder_sentence_count", 0) > 0:
        score += 1.5
    if _is_generic_evidence_title(features.get("title", "")):
        score -= 2.5
    if features.get("false_purpose_sentence_count", 0) > 0 and features.get("purpose_sentence_count", 0) == 0:
        score -= 2.5
    return round(score, 2)


def _display_evidence_score(features):
    score = 0.0
    title = features.get("title", "")
    url = (features.get("url", "") or "").lower()
    score += min(features.get("purpose_sentence_count", 0), 3) * 4.0
    score += min(features.get("stakeholder_sentence_count", 0), 3) * 3.0
    score += min(features.get("leadership_purpose_co_mentions", 0), 3) * 2.5
    score += min(features.get("purpose_strategy_sentences", 0), 3) * 2.0
    score += min(features.get("purpose_capital_sentences", 0), 3) * 1.5
    score += min(features.get("purpose_decision_sentences", 0), 3) * 2.0
    if features.get("source_bucket") == "sec":
        score += 1.2
    if any(marker in url for marker in ["/our-company", "/about", "/purpose", "/mission", "/leadership", "/esg", "/impact", "/credo"]):
        score += 1.5
    if _is_generic_evidence_title(title):
        score -= 4.0
    if any(marker in title.lower() for marker in ["glossary", "gri", "reference architecture", "basepod", "mission control", "benchmarks"]):
        score -= 5.0
    if features.get("false_purpose_sentence_count", 0) > 0:
        score -= 5.0
    if features.get("purpose_sentence_count", 0) == 0 and features.get("stakeholder_sentence_count", 0) == 0:
        score -= 3.0
    return round(score, 2)


def _aggregate_page_features(page_features, company_name):
    summary = {
        "company_name": company_name,
        "record_count": len(page_features),
        "purpose_hits": 0,
        "stakeholder_hits": 0,
        "impact_hits": 0,
        "branding_hits": 0,
        "measurement_hits": 0,
        "purpose_sentence_count": 0,
        "leadership_hits": 0,
        "leadership_purpose_co_mentions": 0,
        "strategy_hits": 0,
        "capital_hits": 0,
        "purpose_strategy_sentences": 0,
        "purpose_capital_sentences": 0,
        "purpose_decision_sentences": 0,
        "employee_hits": 0,
        "employee_embedding_sentences": 0,
        "execution_hits": 0,
        "measurement_hits_execution": 0,
        "capability_hits": 0,
        "industry_hits": 0,
        "years_found": [],
        "sec_filing_years": [],
        "industries_found": [],
        "purpose_examples": [],
        "top_pages": [],
        "source_coverage_count": 0,
        "source_bucket_counts": {},
        "purpose_pages": 0,
        "explicit_purpose_pages": 0,
        "stakeholder_pages": 0,
        "stakeholder_sentence_pages": 0,
        "leadership_pages": 0,
        "strategy_pages": 0,
        "capital_pages": 0,
        "employee_pages": 0,
        "decision_pages": 0,
        "accountability_pages": 0,
        "execution_pages": 0,
        "capability_pages": 0,
        "industry_pages": 0,
        "false_purpose_pages": 0,
        "high_signal_pages": 0,
        "top_page_signal_mean": 0.0,
    }
    year_set = set()
    sec_filing_years = set()
    industry_set = set()
    source_buckets = set()
    purpose_examples = []
    for page in page_features:
        for key in [
            "purpose_hits",
            "stakeholder_hits",
            "impact_hits",
            "branding_hits",
            "measurement_hits",
            "purpose_sentence_count",
            "leadership_hits",
            "leadership_purpose_co_mentions",
            "strategy_hits",
            "capital_hits",
            "purpose_strategy_sentences",
            "purpose_capital_sentences",
            "purpose_decision_sentences",
            "employee_hits",
            "employee_embedding_sentences",
            "execution_hits",
            "measurement_hits_execution",
            "capability_hits",
        ]:
            summary[key] += page.get(key, 0)
        year_set.update(page.get("years_found", []))
        industry_set.update(page.get("industries_found", []))
        source_bucket = page.get("source_bucket", "other")
        source_buckets.add(source_bucket)
        summary["source_bucket_counts"][source_bucket] = summary["source_bucket_counts"].get(source_bucket, 0) + 1
        if source_bucket == "sec":
            sec_filing_years.update(_extract_filing_years(page.get("source_file", "")))
        if page.get("purpose_hits", 0) > 0 or page.get("purpose_sentence_count", 0) > 0:
            summary["purpose_pages"] += 1
        if page.get("purpose_sentence_count", 0) > 0 and page.get("impact_hits", 0) >= 2:
            summary["explicit_purpose_pages"] += 1
        if page.get("stakeholder_hits", 0) > 0:
            summary["stakeholder_pages"] += 1
        if page.get("stakeholder_sentence_count", 0) > 0:
            summary["stakeholder_sentence_pages"] += 1
        if page.get("leadership_purpose_co_mentions", 0) > 0:
            summary["leadership_pages"] += 1
        if page.get("purpose_strategy_sentences", 0) > 0 or page.get("strategy_hits", 0) >= 3:
            summary["strategy_pages"] += 1
        if page.get("purpose_capital_sentences", 0) > 0 or page.get("capital_hits", 0) >= 3:
            summary["capital_pages"] += 1
        if page.get("employee_embedding_sentences", 0) > 0 or page.get("employee_hits", 0) >= 4:
            summary["employee_pages"] += 1
        if page.get("purpose_decision_sentences", 0) > 0:
            summary["decision_pages"] += 1
        if (page.get("measurement_hits", 0) + page.get("measurement_hits_execution", 0)) >= 2:
            summary["accountability_pages"] += 1
        if page.get("execution_hits", 0) >= 3:
            summary["execution_pages"] += 1
        if page.get("capability_hits", 0) >= 3:
            summary["capability_pages"] += 1
        if page.get("industry_hits", 0) >= 1:
            summary["industry_pages"] += 1
        if page.get("false_purpose_sentence_count", 0) > 0 and page.get("purpose_sentence_count", 0) == 0:
            summary["false_purpose_pages"] += 1
        if page.get("page_signal_score", 0) >= 18:
            summary["high_signal_pages"] += 1
        for sentence in page.get("purpose_sentences", []):
            if sentence not in purpose_examples:
                purpose_examples.append(sentence)
    scoring_top_pages = _select_top_pages(page_features)
    display_top_pages = _select_display_pages(page_features)
    summary["industry_hits"] = len(industry_set)
    summary["years_found"] = sorted(year_set)
    summary["sec_filing_years"] = sorted(sec_filing_years)
    summary["industries_found"] = sorted(industry_set)
    summary["purpose_examples"] = _select_display_examples(page_features)
    summary["top_pages"] = [
        {
            "title": page["title"],
            "url": page["url"],
            "display_role": _display_group(page),
            "display_reason": _display_reason(page),
            "page_signal_score": page["page_signal_score"],
            "display_evidence_score": page["display_evidence_score"],
            "purpose_hits": page["purpose_hits"],
            "impact_hits": page["impact_hits"],
            "strategy_hits": page["strategy_hits"],
        }
        for page in display_top_pages
    ]
    summary["scoring_top_pages"] = [
        {
            "title": page["title"],
            "url": page["url"],
            "page_signal_score": page["page_signal_score"],
            "display_evidence_score": page["display_evidence_score"],
            "purpose_hits": page["purpose_hits"],
            "impact_hits": page["impact_hits"],
            "strategy_hits": page["strategy_hits"],
        }
        for page in scoring_top_pages
    ]
    summary["source_coverage_count"] = len(source_buckets)
    summary["top_page_signal_mean"] = round(
        safe_ratio(sum(page["page_signal_score"] for page in scoring_top_pages), len(scoring_top_pages)),
        2,
    )
    for prefix in [
        "purpose",
        "explicit_purpose",
        "stakeholder",
        "stakeholder_sentence",
        "leadership",
        "strategy",
        "capital",
        "employee",
        "decision",
        "accountability",
        "execution",
        "capability",
        "industry",
        "false_purpose",
        "high_signal",
    ]:
        summary[f"{prefix}_page_ratio"] = round(safe_ratio(summary[f"{prefix}_pages"], summary["record_count"]), 4)
    summary["sec_year_coverage_ratio"] = round(
        safe_ratio(len([year for year in summary["sec_filing_years"] if 2021 <= year <= 2025]), 5),
        4,
    )
    return summary


def _select_top_pages(page_features):
    selected = []
    seen_titles = set()
    seen_buckets = set()
    sorted_pages = sorted(page_features, key=lambda item: item["page_signal_score"], reverse=True)
    for page in sorted_pages:
        title = page.get("title", "").strip().lower()
        if title in seen_titles:
            continue
        if _is_generic_evidence_title(page.get("title", "")) and page.get("purpose_sentence_count", 0) == 0:
            continue
        if page.get("false_purpose_sentence_count", 0) > 0 and page.get("purpose_sentence_count", 0) == 0:
            continue
        bucket = page.get("source_bucket", "other")
        if bucket in seen_buckets and len(selected) < 3 and page.get("purpose_sentence_count", 0) == 0:
            continue
        selected.append(page)
        seen_titles.add(title)
        seen_buckets.add(bucket)
        if len(selected) == 5:
            break
    if not selected:
        selected = sorted_pages[:5]
    return selected


def _select_display_pages(page_features):
    selected = []
    selected_titles = set()
    ranked_pages = sorted(page_features, key=_display_rank_key, reverse=True)
    preferred_roles = ["purpose", "leadership", "impact", "people", "strategy"]

    for role in preferred_roles:
        candidate = _pick_display_page(
            ranked_pages,
            selected_titles,
            required_roles={role},
        )
        if candidate:
            selected.append(candidate)
            selected_titles.add(candidate.get("title", "").strip().lower())

    while len(selected) < DEFAULT_TOP_EVIDENCE_COUNT:
        candidate = _pick_display_page(
            ranked_pages,
            selected_titles,
            allowed_roles={"purpose", "leadership", "impact", "people", "strategy", "about", "execution"},
        )
        if not candidate:
            break
        selected.append(candidate)
        selected_titles.add(candidate.get("title", "").strip().lower())

    if not selected:
        return _select_top_pages(page_features)
    return selected


def _display_group(page):
    title = (page.get("title", "") or "").lower()
    url = (page.get("url", "") or "").lower()
    if any(marker in title or marker in url for marker in ["careers", "people", "culture", "benefits"]):
        return "people"
    if any(marker in title or marker in url for marker in ["foundation", "ai for good"]):
        return "impact"
    if any(marker in title or marker in url for marker in ["credo", "purpose", "mission", "our-company", "/about"]):
        return "purpose"
    if any(marker in title or marker in url for marker in ["leadership", "ceo", "chair", "board"]):
        return "leadership"
    if any(marker in title or marker in url for marker in ["esg", "impact", "sustainability", "social-impact"]):
        return "impact"
    if any(marker in title or marker in url for marker in ["corporate", "company information", "our company", "/corporate/"]):
        return "about"
    if any(marker in title or marker in url for marker in ["strategy", "governance", "invest", "report", "10k"]):
        return "strategy"
    if (
        page.get("purpose_decision_sentences", 0) > 0
        or page.get("purpose_capital_sentences", 0) > 0
        or page.get("execution_hits", 0) >= 4
    ):
        return "execution"
    return "other"


def _select_purpose_examples(purpose_examples):
    cleaned = []
    for sentence in purpose_examples:
        text = sentence.strip()
        lowered = text.lower()
        if not text:
            continue
        if len(text.split()) < 8:
            continue
        if _looks_like_navigation_text(lowered):
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned[:5]


def _select_display_examples(page_features):
    examples = []
    ranked_pages = _select_display_pages(page_features)
    for page in ranked_pages:
        for sentence in page.get("human_evidence_sentences", []) + page.get("purpose_sentences", []):
            cleaned = _clean_display_sentence(sentence)
            if not cleaned:
                continue
            if cleaned not in examples:
                examples.append(cleaned)
        if len(examples) >= 5:
            break
    return examples[:5]


def _is_generic_evidence_title(title):
    lowered = str(title).lower()
    return any(pattern in lowered for pattern in GENERIC_EVIDENCE_TITLE_PATTERNS)


def _is_display_excluded(page):
    title = str(page.get("title", "")).lower()
    url = str(page.get("url", "")).lower()
    if page.get("source_bucket") == "sec":
        return True
    if any(pattern in title for pattern in DISPLAY_EXCLUDE_TITLE_PATTERNS):
        return True
    if any(pattern in url for pattern in DISPLAY_EXCLUDE_URL_PATTERNS):
        return True
    if any(
        marker in title
        for marker in [
            "partner network",
            "personal time off",
            "pto and holidays",
            "training",
            "military and veterans",
            "support jobs",
            "accessibility policies",
            "legal notices",
            "brand guidelines",
            "trademarks, licenses",
        ]
    ):
        return True
    if any(
        marker in url
        for marker in [
            "/partners",
            "/parental-leave",
            "/culture/training",
            "/life-at-oracle/veterans",
            "/careers/opportunities/",
            "/accessibility/policy",
            "/terms-of-service",
            "/logo-brand-usage",
        ]
    ):
        return True
    return False


def _display_rank_key(page):
    role = _display_group(page)
    base_score = float(page.get("display_evidence_score", 0))
    title = (page.get("title", "") or "").lower()
    url = (page.get("url", "") or "").lower()
    role_bonus = {
        "purpose": 6.0,
        "leadership": 4.5,
        "impact": 4.0,
        "about": 3.5,
        "strategy": 3.0,
        "people": 2.5,
        "execution": 1.5,
        "other": 0.0,
    }.get(role, 0.0)
    quality_bonus = 0.0
    if page.get("purpose_sentence_count", 0) > 0:
        quality_bonus += 4.0
    if page.get("stakeholder_sentence_count", 0) > 0:
        quality_bonus += 2.5
    if page.get("leadership_purpose_co_mentions", 0) > 0:
        quality_bonus += 1.5
    if any(marker in url for marker in ["/about", "/corporate", "/social-impact", "/foundation", "/leadership-message"]):
        quality_bonus += 2.0
    if any(marker in title for marker in ["executive biography", "benefits", "career development", "military and veterans"]):
        quality_bonus -= 2.0
    return (base_score + role_bonus + quality_bonus, page.get("page_signal_score", 0))


def _pick_display_page(ranked_pages, selected_titles, required_roles=None, allowed_roles=None):
    for page in ranked_pages:
        title = page.get("title", "").strip().lower()
        if title in selected_titles:
            continue
        if page.get("display_evidence_score", 0) <= 0:
            continue
        if _is_display_excluded(page):
            continue
        if not _is_human_evidence_candidate(page):
            continue
        role = _display_group(page)
        if required_roles and role not in required_roles:
            continue
        if allowed_roles and role not in allowed_roles:
            continue
        if not _meets_display_role_threshold(page, role):
            continue
        return page
    return None


def _is_human_evidence_candidate(page):
    role = _display_group(page)
    if page.get("purpose_sentence_count", 0) > 0:
        return True
    if page.get("stakeholder_sentence_count", 0) > 0 and role in {"purpose", "leadership", "impact", "about"}:
        return True
    if page.get("leadership_purpose_co_mentions", 0) > 0 and role in {"leadership", "impact", "about", "strategy"}:
        return True
    if page.get("employee_embedding_sentences", 0) > 0 and role == "people":
        return True
    if page.get("purpose_strategy_sentences", 0) > 0 and role in {"strategy", "impact", "about"}:
        return True
    if page.get("purpose_capital_sentences", 0) > 0 and role in {"strategy", "impact"}:
        return True
    return False


def _display_reason(page):
    reasons = []
    if page.get("purpose_sentence_count", 0) > 0:
        reasons.append("explicit purpose language")
    if page.get("stakeholder_sentence_count", 0) > 0:
        reasons.append("stakeholder framing")
    if page.get("leadership_purpose_co_mentions", 0) > 0:
        reasons.append("leadership linkage")
    if page.get("purpose_strategy_sentences", 0) > 0:
        reasons.append("strategy linkage")
    if page.get("employee_embedding_sentences", 0) > 0:
        reasons.append("people embedding")
    return ", ".join(reasons[:3]) or "contextual evidence"


def _meets_display_role_threshold(page, role):
    if role == "purpose":
        return page.get("purpose_sentence_count", 0) > 0
    if role == "leadership":
        return page.get("leadership_purpose_co_mentions", 0) > 0 or page.get("purpose_sentence_count", 0) > 0
    if role == "impact":
        return page.get("stakeholder_sentence_count", 0) > 0 or page.get("purpose_sentence_count", 0) > 0
    if role == "people":
        return page.get("employee_embedding_sentences", 0) > 0 or page.get("stakeholder_sentence_count", 0) > 0
    if role == "strategy":
        return page.get("purpose_strategy_sentences", 0) > 0 or page.get("purpose_capital_sentences", 0) > 0
    if role == "execution":
        return (
            page.get("purpose_decision_sentences", 0) > 0
            or page.get("purpose_capital_sentences", 0) > 0
            or page.get("purpose_strategy_sentences", 0) > 0
        )
    return True


def _clean_display_sentence(sentence):
    text = " ".join(str(sentence or "").split())
    lowered = text.lower()
    if not text:
        return ""
    if len(text.split()) < 8 or len(text) > 320:
        return ""
    if _looks_like_navigation_text(lowered):
        return ""
    if any(
        marker in lowered
        for marker in [
            "purpose-built",
            "benchmark",
            "documentation overview",
            "new releases",
            "executive biography",
            "joyce westerdahl",
            "chief executive officer",
        ]
    ):
        return ""
    if text.count("|") >= 2:
        return ""
    return text


def _looks_like_navigation_text(text):
    bad_markers = [
        "skip to main",
        "skip to footer",
        " menu ",
        "page overview",
        "featured stories",
        "how can i help",
        "policies contact",
    ]
    pipe_count = text.count("|")
    if pipe_count >= 4:
        return True
    return any(marker in text for marker in bad_markers)


def _classify_source(source_file):
    lowered = str(source_file).lower()
    if "linkedin_selected_10_companies" in lowered:
        return "linkedin"
    if "\\10ks\\" in lowered or "/10ks/" in lowered or "_10k_" in lowered:
        return "sec"
    if "selected_10_companies" in lowered:
        return "web"
    return "other"


def _extract_filing_years(source_file):
    name = Path(str(source_file)).name
    if len(name) >= 4 and name[:4].isdigit():
        return [int(name[:4])]
    return []


def _page_rows(page_features):
    rows = []
    for page in sorted(page_features, key=lambda item: item["page_signal_score"], reverse=True):
        rows.append(
            {
                "title": page["title"],
                "url": page["url"],
                "page_signal_score": page["page_signal_score"],
                "purpose_hits": page["purpose_hits"],
                "impact_hits": page["impact_hits"],
                "leadership_purpose_co_mentions": page["leadership_purpose_co_mentions"],
                "strategy_hits": page["strategy_hits"],
                "capital_hits": page["capital_hits"],
                "employee_hits": page["employee_hits"],
                "execution_hits": page["execution_hits"],
                "capability_hits": page["capability_hits"],
                "industry_hits": page["industry_hits"],
                "source_file": page["source_file"],
            }
        )
    return rows


if __name__ == "__main__":
    main()
