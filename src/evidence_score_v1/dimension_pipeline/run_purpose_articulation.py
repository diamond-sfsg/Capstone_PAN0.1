from __future__ import annotations

from pathlib import Path
import sys

# Make project root importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_score_v1.config import PHASE2_OUTPUT_ROOT, PURPOSE_OUTPUT_COLUMNS
from src.evidence_score_v1.dimension_config import PURPOSE_ARTICULATION
from src.evidence_score_v1.data_loader import load_chunk_corpus, select_base_columns
from src.evidence_score_v1.filters import add_retrieval_status, filter_retrieval_rows
from src.evidence_score_v1.lexical_score import add_lexical_scores
from src.evidence_score_v1.tfidf_score import compute_tfidf_scores
from src.evidence_score_v1.embedding_score import compute_embedding_scores
from src.evidence_score_v1.metadata_score import add_metadata_scores
from src.evidence_score_v1.ranker import add_score_ranks
from src.evidence_score_v1.exporter import export_scores_csv
from src.evidence_score_v1.report import build_score_report, export_report


def main() -> None:
    output_csv = PHASE2_OUTPUT_ROOT / "purpose_articulation_scores_v1.csv"
    output_report = PHASE2_OUTPUT_ROOT / "purpose_articulation_report_v1.txt"

    # 1. Load
    df = load_chunk_corpus()
    df = select_base_columns(df)

    # 2. Filter
    df = add_retrieval_status(df)
    df_kept = filter_retrieval_rows(df)

    # 3. Lexical
    df_kept = add_lexical_scores(
        df_kept,
        keywords_core=PURPOSE_ARTICULATION["keywords_core"],
        keywords_support=PURPOSE_ARTICULATION["keywords_support"],
        text_col="text_clean",
    )

    # 4. TF-IDF
    df_kept = compute_tfidf_scores(
        df_kept,
        query_text=PURPOSE_ARTICULATION["query_text"],
        text_col="text_clean",
    )

    # 5. Embedding
    df_kept = compute_embedding_scores(
        df_kept,
        query_text=PURPOSE_ARTICULATION["query_text"],
        text_col="text_clean",
    )

    # 6. Metadata
    df_kept = add_metadata_scores(
        df_kept,
        preferred_sections=PURPOSE_ARTICULATION["preferred_sections"],
        preferred_sources=PURPOSE_ARTICULATION["preferred_sources"],
    )

    # 7. Ranking
    df_kept = add_score_ranks(df_kept)

    # 8. Reorder columns
    final_cols = [c for c in PURPOSE_OUTPUT_COLUMNS if c in df_kept.columns]
    df_out = df_kept[final_cols].copy()

    # 9. Export
    export_scores_csv(df_out, output_csv)

    report_text = build_score_report(df_all=df, df_kept=df_out)
    export_report(report_text, output_report)

    print(f"Saved scores to: {output_csv}")
    print(f"Saved report to: {output_report}")
    print(f"Kept rows: {len(df_out)}")


if __name__ == "__main__":
    main()