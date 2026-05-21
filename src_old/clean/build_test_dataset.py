from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "clean_2.0" / "unified_chunks_final_v4.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "test"
SOURCE_COLUMNS = ["edgar", "official_web", "linkedin"]


def build_company_summary(input_csv: Path) -> pd.DataFrame:
    usecols = ["company", "year", "source"]
    parts = []
    year_pairs = []

    for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=50_000):
        chunk["company"] = chunk["company"].fillna("").astype(str).str.strip()
        chunk["source"] = chunk["source"].fillna("unknown").astype(str).str.strip()
        chunk = chunk[chunk["company"] != ""].copy()

        if chunk.empty:
            continue

        valid_years = chunk[["company", "year"]].dropna().drop_duplicates()
        if not valid_years.empty:
            year_pairs.append(valid_years)

        base = chunk.groupby("company").agg(
            chunk_count=("company", "size"),
        )
        source_counts = pd.crosstab(chunk["company"], chunk["source"])
        parts.append(base.join(source_counts, how="left"))

    if not parts:
        raise ValueError(f"No company rows found in {input_csv}.")

    summary = pd.concat(parts).groupby(level=0).sum(numeric_only=True)

    for source in SOURCE_COLUMNS:
        if source not in summary.columns:
            summary[source] = 0

    summary["source_count"] = (summary[SOURCE_COLUMNS] > 0).sum(axis=1)
    summary["year_count"] = 0
    summary["min_year"] = pd.NA
    summary["max_year"] = pd.NA

    if year_pairs:
        years = pd.concat(year_pairs, ignore_index=True).drop_duplicates()
        year_stats = years.groupby("company")["year"].agg(
            year_count="nunique",
            min_year="min",
            max_year="max",
        )
        summary.update(year_stats)

    summary["other_source_count"] = summary["official_web"] + summary["linkedin"]
    summary["three_source_min"] = summary[SOURCE_COLUMNS].min(axis=1)
    summary["three_source_max"] = summary[SOURCE_COLUMNS].max(axis=1)
    summary["three_source_gap"] = summary["three_source_max"] - summary["three_source_min"]
    summary["three_source_balance_ratio"] = (
        summary["three_source_min"] / summary["three_source_max"].replace(0, pd.NA)
    ).fillna(0)
    summary["edgar_other_gap"] = (summary["edgar"] - summary["other_source_count"]).abs()
    summary["ticker"] = summary.index
    summary["company"] = summary.index

    return summary.reset_index(drop=True)


def _split_high_low(
    coverage: pd.DataFrame,
    n_companies: int,
    high_sort: list[str],
    high_ascending: list[bool],
    low_sort: list[str],
    low_ascending: list[bool],
    high_label: str,
    low_label: str,
) -> pd.DataFrame:
    if n_companies < 2:
        raise ValueError("n_companies must be at least 2.")
    if len(coverage) < n_companies:
        raise ValueError(f"Only {len(coverage)} companies found, cannot select {n_companies}.")

    high_n = n_companies // 2
    low_n = n_companies - high_n

    high = coverage.sort_values(high_sort, ascending=high_ascending).head(high_n)
    high = high.assign(selection_group=high_label)

    remaining = coverage[~coverage["company"].isin(high["company"])]
    low = remaining.sort_values(low_sort, ascending=low_ascending).head(low_n)
    low = low.assign(selection_group=low_label)

    return pd.concat([high, low], ignore_index=True)


def select_test1_num(summary: pd.DataFrame, n_companies: int) -> pd.DataFrame:
    selected = _split_high_low(
        summary,
        n_companies,
        high_sort=["chunk_count", "source_count", "year_count", "company"],
        high_ascending=[False, False, False, True],
        low_sort=["chunk_count", "source_count", "year_count", "company"],
        low_ascending=[True, True, True, True],
        high_label="max_data_volume",
        low_label="min_data_volume",
    )
    return selected


def select_test2_year(summary: pd.DataFrame, n_companies: int) -> pd.DataFrame:
    selected = _split_high_low(
        summary,
        n_companies,
        high_sort=["year_count", "chunk_count", "source_count", "company"],
        high_ascending=[False, False, False, True],
        low_sort=["year_count", "chunk_count", "source_count", "company"],
        low_ascending=[True, True, True, True],
        high_label="max_year_coverage",
        low_label="min_year_coverage",
    )
    return selected


def select_test3_source(summary: pd.DataFrame, n_companies: int) -> pd.DataFrame:
    eligible = summary[
        (summary["edgar"] > 0)
        & (summary["official_web"] > 0)
        & (summary["linkedin"] > 0)
    ].copy()

    selected = _split_high_low(
        eligible,
        n_companies,
        high_sort=["three_source_balance_ratio", "three_source_gap", "chunk_count", "company"],
        high_ascending=[False, True, False, True],
        low_sort=["three_source_balance_ratio", "three_source_gap", "chunk_count", "company"],
        low_ascending=[True, False, False, True],
        high_label="most_source_balanced",
        low_label="largest_source_gap",
    )
    return selected


def selected_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "company",
        "selection_group",
        "chunk_count",
        "source_count",
        "year_count",
        "min_year",
        "max_year",
        "edgar",
        "official_web",
        "linkedin",
        "other_source_count",
        "three_source_gap",
        "three_source_balance_ratio",
        "edgar_other_gap",
    ]
    return df[columns].copy()


def export_test_dataset(input_csv: Path, output_dir: Path, selected: pd.DataFrame, test_name: str) -> Path:
    tickers = set(selected["ticker"])
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for chunk in pd.read_csv(input_csv, chunksize=50_000, low_memory=False):
        subset = chunk[chunk["company"].isin(tickers)].copy()
        if not subset.empty:
            chunks.append(subset)

    if not chunks:
        raise ValueError(f"No chunks matched selected tickers for {test_name}.")

    test_df = pd.concat(chunks, ignore_index=True)
    out_csv = output_dir / f"unified_chunks_{test_name}_v4.csv"
    test_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


def write_outputs(
    output_dir: Path,
    test_name: str,
    selected: pd.DataFrame,
    test_csv: Path,
    source_csv: Path,
    selection_rule: str,
) -> None:
    ticker_csv = output_dir / f"{test_name}_ticker_v4.csv"
    manifest_json = output_dir / f"{test_name}_manifest_v4.json"

    selected.to_csv(ticker_csv, index=False, encoding="utf-8-sig")
    manifest_selected = selected.astype(object).where(pd.notna(selected), None)

    manifest = {
        "test_name": test_name,
        "source_csv": str(source_csv),
        "test_csv": str(test_csv),
        "test_ticker_csv": str(ticker_csv),
        "selection_rule": selection_rule,
        "test_ticker": selected["ticker"].tolist(),
        "selected_companies": manifest_selected.to_dict(orient="records"),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


def build_one_test(
    *,
    input_csv: Path,
    output_dir: Path,
    test_name: str,
    selected: pd.DataFrame,
    selection_rule: str,
) -> None:
    selected = selected_columns(selected)
    test_csv = export_test_dataset(input_csv, output_dir, selected, test_name)
    write_outputs(output_dir, test_name, selected, test_csv, input_csv, selection_rule)

    print("")
    print(f"{test_name}_ticker =", selected["ticker"].tolist())
    print(f"ticker_csv = {output_dir / f'{test_name}_ticker_v4.csv'}")
    print(f"test_csv   = {test_csv}")
    print(selected.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build named v4 test datasets from unified ticker-indexed chunks."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-companies", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    n_companies = args.n_companies

    summary = build_company_summary(input_csv)

    tests = [
        (
            "test1_num",
            select_test1_num(summary, n_companies),
            "5 tickers with maximum final chunk volume and 5 tickers with minimum final chunk volume.",
        ),
        (
            "test2_year",
            select_test2_year(summary, n_companies),
            "5 tickers with maximum distinct-year coverage and 5 tickers with minimum distinct-year coverage.",
        ),
        (
            "test3_source",
            select_test3_source(summary, n_companies),
            "Among tickers with edgar, official_web, and linkedin data: 5 with the most balanced source mix and 5 with the largest source mix gap.",
        ),
    ]

    for test_name, selected, selection_rule in tests:
        build_one_test(
            input_csv=input_csv,
            output_dir=output_dir,
            test_name=test_name,
            selected=selected,
            selection_rule=selection_rule,
        )


if __name__ == "__main__":
    main()
