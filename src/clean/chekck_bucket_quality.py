from pathlib import Path
import re

import pandas as pd


INPUT_CSV = Path("data/clean/unified_corpus_bucketed.csv")
REPORT_PATH = Path("data/clean/bucket_quality_report.txt")
SAMPLE_DIR = Path("data/clean/bucket_samples")


NOISE_PATTERNS = [
    r"trending news and stories",
    r"related tags share",
    r"you might also like",
    r"read more",
    r"learn more",
    r"cookie",
    r"privacy policy",
    r"sign in",
    r"join now",
    r"page overview",
]


def clean_len(text: str) -> int:
    if pd.isna(text) or text is None:
        return 0
    return len(str(text))


def detect_noise(text: str) -> int:
    if pd.isna(text) or text is None:
        return 0
    text = str(text).lower()
    return sum(1 for p in NOISE_PATTERNS if re.search(p, text))


def write_report(df: pd.DataFrame):
    lines = []

    lines.append("BUCKET QUALITY REPORT")
    lines.append("=" * 80)
    lines.append(f"Total rows: {len(df)}")
    lines.append("")

    lines.append("1. PRIMARY BUCKET DISTRIBUTION")
    lines.append("-" * 80)
    lines.append(df["primary_bucket"].value_counts(dropna=False).to_string())
    lines.append("")

    lines.append("2. PRIMARY BUCKET x SOURCE")
    lines.append("-" * 80)
    lines.append(pd.crosstab(df["primary_bucket"], df["source"], dropna=False).to_string())
    lines.append("")

    lines.append("3. PRIMARY BUCKET x COMPANY")
    lines.append("-" * 80)
    for bucket in sorted(df["primary_bucket"].dropna().unique()):
        lines.append(f"[{bucket}]")
        subset = df[df["primary_bucket"] == bucket]
        lines.append(subset["company"].value_counts().head(15).to_string())
        lines.append("")

    lines.append("4. LENGTH STATS BY PRIMARY BUCKET")
    lines.append("-" * 80)
    length_stats = (
        df.groupby("primary_bucket")["text_len"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )
    lines.append(length_stats.to_string())
    lines.append("")

    lines.append("5. SHORT TEXT RATES BY PRIMARY BUCKET (text_len < 200)")
    lines.append("-" * 80)
    short_stats = (
        df.assign(is_short=df["text_len"] < 200)
        .groupby("primary_bucket")["is_short"]
        .mean()
        .sort_values(ascending=False)
        .round(4)
    )
    lines.append(short_stats.to_string())
    lines.append("")

    lines.append("6. NOISE HITS BY PRIMARY BUCKET")
    lines.append("-" * 80)
    noise_stats = (
        df.groupby("primary_bucket")["noise_hit_count"]
        .agg(["mean", "max", "sum"])
        .round(2)
    )
    lines.append(noise_stats.to_string())
    lines.append("")

    lines.append("7. LOW-CONFIDENCE RECORDS (bucket_confidence < 0.20)")
    lines.append("-" * 80)
    low_conf = df[df["bucket_confidence"].fillna(0) < 0.20]
    lines.append(f"Count: {len(low_conf)}")
    if len(low_conf) > 0:
        lines.append(
            low_conf[["company", "source", "doc_type", "primary_bucket", "secondary_bucket", "bucket_confidence"]]
            .head(30)
            .to_string(index=False)
        )
    lines.append("")

    lines.append("8. POTENTIAL SOURCE-BUCKET MISMATCHES")
    lines.append("-" * 80)

    mismatches = []

    # heuristics for suspicious combinations
    suspicious_1 = df[
        (df["primary_bucket"] == "execution_outcome_impact") &
        (df["source"] == "linkedin")
    ]
    mismatches.append(("execution_outcome_impact from linkedin", suspicious_1))

    suspicious_2 = df[
        (df["primary_bucket"] == "purpose_articulation") &
        (df["source"] == "edgar") &
        (df["doc_type"] == "risk")
    ]
    mismatches.append(("purpose_articulation from edgar risk", suspicious_2))

    suspicious_3 = df[
        (df["primary_bucket"] == "organizational_alignment") &
        (df["source"] == "edgar") &
        (~df["doc_type"].fillna("").isin(["business", "mdna"]))
    ]
    mismatches.append(("organizational_alignment from unlikely edgar docs", suspicious_3))

    for label, subset in mismatches:
        lines.append(f"{label}: {len(subset)}")
        if len(subset) > 0:
            lines.append(
                subset[["company", "source", "doc_type", "page_title", "bucket_confidence"]]
                .head(10)
                .to_string(index=False)
            )
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Report saved to {REPORT_PATH}")


def export_samples(df: pd.DataFrame, n_per_bucket: int = 25):
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for bucket in sorted(df["primary_bucket"].dropna().unique()):
        subset = df[df["primary_bucket"] == bucket].copy()

        if subset.empty:
            continue

        sample_size = min(n_per_bucket, len(subset))
        sample_df = subset.sample(sample_size, random_state=42)

        out_path = SAMPLE_DIR / f"{bucket}_sample.csv"
        sample_df[
            [
                "chunk_id", "company", "source", "doc_type",
                "primary_bucket", "secondary_bucket", "bucket_confidence",
                "page_title", "url", "text"
            ]
        ].to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"[DONE] Sample exported: {out_path} ({sample_size} rows)")


def run():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Loaded {len(df)} rows from {INPUT_CSV}")

    df["text_len"] = df["text"].apply(clean_len)
    df["noise_hit_count"] = df["text"].apply(detect_noise)

    print("\n===== BUCKET x SOURCE =====")
    print(pd.crosstab(df["primary_bucket"], df["source"], dropna=False))

    print("\n===== LENGTH STATS =====")
    print(
        df.groupby("primary_bucket")["text_len"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )

    write_report(df)
    export_samples(df, n_per_bucket=25)


if __name__ == "__main__":
    run()