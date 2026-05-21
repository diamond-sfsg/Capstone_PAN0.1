import json
import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


INPUT_CSV = Path("data/clean/unified_corpus.csv")
OUTPUT_CSV = Path("data/clean/unified_corpus_bucketed.csv")
OUTPUT_DB = Path("data/clean/unified_corpus_bucketed.db")


BUCKETS = [
    "purpose_articulation",
    "history_consistency",
    "strategy_resource_alignment",
    "organizational_alignment",
    "execution_outcome_impact",
]


PURPOSE_KEYWORDS = [
    "purpose", "mission", "values", "why we exist", "our purpose",
    "our mission", "long-term thinking", "commitment", "believe",
    "for our customers", "for our communities", "force for good",
    "responsibility", "vision"
]

HISTORY_KEYWORDS = [
    "since", "over the years", "year-over-year", "long-term",
    "continued", "continue to", "progress since", "from 20",
    "by 20", "over time", "years", "history", "evolution",
    "tracking progress", "annual progress", "timeline"
]

STRATEGY_KEYWORDS = [
    "strategy", "strategic", "invest", "investment", "investing",
    "committed", "commitment", "allocate", "allocation", "capital",
    "fund", "initiative", "procurement", "infrastructure",
    "plan", "roadmap", "scale", "expand", "expansion",
    "target", "goal", "program", "portfolio"
]

ORG_KEYWORDS = [
    "leadership", "leaders", "leadership principles", "culture",
    "employees", "workforce", "governance", "board", "management",
    "training", "upskilling", "career", "talent", "hiring",
    "diversity", "inclusion", "employer", "safety", "workplace",
    "principles", "accountability", "organization", "organizational"
]

EXECUTION_KEYWORDS = [
    "results", "achieved", "delivered", "metrics", "outcome",
    "outcomes", "report", "reported", "reduced", "increased",
    "improved", "progress", "impact", "generated", "created",
    "preserved", "million", "billion", "percent", "%",
    "kpi", "recordable incident rate", "ltir", "rir",
    "homes", "jobs", "meals", "gigawatts", "hectares"
]


def normalize_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_score(text: str, keywords: list[str]) -> int:
    score = 0
    for kw in keywords:
        if kw in text:
            score += 1
    return score


def has_many_numbers(text: str) -> bool:
    number_hits = re.findall(r"\b\d[\d,.\-]*\b|%", text)
    return len(number_hits) >= 3


def get_source_prior(source: str, doc_type: str) -> dict:
    """
    Source/doc_type priors help coarse routing.
    """
    priors = {b: 0 for b in BUCKETS}

    source = (source or "").lower()
    doc_type = (doc_type or "").lower()

    if source == "edgar":
        priors["strategy_resource_alignment"] += 2
        priors["execution_outcome_impact"] += 2
        priors["history_consistency"] += 1

    if source == "linkedin":
        priors["purpose_articulation"] += 1
        priors["organizational_alignment"] += 2

    if source == "official_web":
        priors["purpose_articulation"] += 2
        priors["strategy_resource_alignment"] += 1
        priors["organizational_alignment"] += 1
        priors["execution_outcome_impact"] += 1

    if doc_type in {"mission_page", "about_page", "company_about"}:
        priors["purpose_articulation"] += 3

    if doc_type in {"leadership_page", "governance_page"}:
        priors["organizational_alignment"] += 3

    if doc_type in {"investor_page", "sustainability_page", "impact_page"}:
        priors["strategy_resource_alignment"] += 2
        priors["execution_outcome_impact"] += 2
        priors["history_consistency"] += 1

    if doc_type in {"business", "mdna"}:
        priors["strategy_resource_alignment"] += 2
        priors["execution_outcome_impact"] += 2

    if doc_type == "risk":
        priors["strategy_resource_alignment"] += 1
        priors["execution_outcome_impact"] += 1

    return priors


def score_buckets(row: pd.Series) -> dict:
    text = normalize_text(row.get("text", ""))
    title = normalize_text(row.get("page_title", ""))
    source = normalize_text(row.get("source", ""))
    doc_type = normalize_text(row.get("doc_type", ""))
    section = normalize_text(row.get("section", ""))
    full_text = " ".join([title, section, text[:2500]])

    scores = {b: 0 for b in BUCKETS}

    priors = get_source_prior(source, doc_type)
    for b, v in priors.items():
        scores[b] += v

    scores["purpose_articulation"] += keyword_score(full_text, PURPOSE_KEYWORDS)
    scores["history_consistency"] += keyword_score(full_text, HISTORY_KEYWORDS)
    scores["strategy_resource_alignment"] += keyword_score(full_text, STRATEGY_KEYWORDS)
    scores["organizational_alignment"] += keyword_score(full_text, ORG_KEYWORDS)
    scores["execution_outcome_impact"] += keyword_score(full_text, EXECUTION_KEYWORDS)

    # heuristics
    if has_many_numbers(full_text):
        scores["execution_outcome_impact"] += 2

    if "leadership principles" in full_text:
        scores["organizational_alignment"] += 4

    if "net-zero" in full_text or "climate pledge" in full_text:
        scores["strategy_resource_alignment"] += 2
        scores["execution_outcome_impact"] += 1
        scores["purpose_articulation"] += 1

    if "year-over-year" in full_text or "since 20" in full_text:
        scores["history_consistency"] += 3

    return scores


def pick_buckets(scores: dict) -> Tuple[Optional[str], Optional[str], float]:
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    primary_bucket, primary_score = sorted_items[0]
    secondary_bucket, secondary_score = sorted_items[1]

    if primary_score <= 0:
        return None, None, 0.0

    # keep secondary only if close enough to primary
    if secondary_score < max(2, primary_score - 2):
        secondary_bucket = None

    confidence = round(primary_score / max(1, sum(scores.values())), 4)
    return primary_bucket, secondary_bucket, confidence


def assign_bucket_row(row: pd.Series) -> pd.Series:
    scores = score_buckets(row)
    primary_bucket, secondary_bucket, confidence = pick_buckets(scores)

    row["primary_bucket"] = primary_bucket
    row["secondary_bucket"] = secondary_bucket
    row["bucket_confidence"] = confidence
    row["bucket_method"] = "rule_v1"
    row["bucket_scores_json"] = json.dumps(scores, ensure_ascii=False)
    return row


def save_sqlite(df: pd.DataFrame):
    conn = sqlite3.connect(OUTPUT_DB)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS bucketed_documents")

    cur.execute("""
    CREATE TABLE bucketed_documents (
        doc_id TEXT,
        chunk_id TEXT PRIMARY KEY,
        company TEXT,
        ticker TEXT,
        year INTEGER,
        source TEXT,
        source_type TEXT,
        doc_type TEXT,
        section TEXT,
        page_title TEXT,
        url TEXT,
        text TEXT,
        metadata_json TEXT,
        primary_bucket TEXT,
        secondary_bucket TEXT,
        bucket_confidence REAL,
        bucket_method TEXT,
        bucket_scores_json TEXT
    )
    """)

    df.to_sql("bucketed_documents", conn, if_exists="append", index=False)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_primary_bucket ON bucketed_documents(primary_bucket)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_company ON bucketed_documents(company)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON bucketed_documents(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_type ON bucketed_documents(doc_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_year ON bucketed_documents(year)")

    conn.commit()
    conn.close()

    print(f"[DONE] SQLite saved to {OUTPUT_DB}")


def run():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Loaded {len(df)} rows from {INPUT_CSV}")

    df = df.apply(assign_bucket_row, axis=1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] Bucketed CSV saved to {OUTPUT_CSV}")

    save_sqlite(df)

    print("\n===== PRIMARY BUCKET DISTRIBUTION =====")
    print(df["primary_bucket"].value_counts(dropna=False))

    print("\n===== PRIMARY BUCKET x SOURCE =====")
    print(pd.crosstab(df["primary_bucket"], df["source"], dropna=False))


if __name__ == "__main__":
    run()