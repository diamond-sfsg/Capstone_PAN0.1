# Evidence Phase 2-3 Pipeline

This document describes the current evidence pipeline from Phase 2 scoring through Phase 3 LLM review.

The current pipeline uses the new data source:

```text
data/clean_2.0/unified_chunks_v4.csv
```

All new-data outputs use the `_newdata` suffix so they do not overwrite older `v1` outputs.

## Pipeline Overview

```text
Phase 2A: evidence scoring
  input : data/clean_2.0/unified_chunks_v4.csv
  script: src/evidence_score_v1/pipeline/run_evidence_score.py
  output: data/phase2/evidence_score_v1_newdata.csv

Phase 2B: evidence bucket assignment
  input : data/phase2/evidence_score_v1_newdata.csv
  script: src/evidence_buckets_v1/run_evidence_buckets.py
  output: data/phase2/evidence_buckets_v1/*_newdata.csv

Phase 3: LLM evidence review
  input : data/phase2/evidence_buckets_v1/*_evidence_v1_newdata.csv
  script: src/evidence_llm_review_v1/run_llm_evidence_review.py
  output: data/phase3/evidence_llm_review_v1/*_newdata.csv
```

## Phase 2A: Evidence Scoring

Run from the project root:

```powershell
python src\evidence_score_v1\pipeline\run_evidence_score.py
```

Current config:

```text
src/evidence_score_v1/config.py
```

Important paths:

```text
DEFAULT_INPUT_CSV           = data/clean_2.0/unified_chunks_v4.csv
EVIDENCE_MATRIX_OUTPUT_PATH = data/phase2/evidence_score_v1_newdata.csv
DEFAULT_DIAGNOSTICS_TXT     = data/phase2/evidence_score_v1_diagnostics_newdata.txt
```

The scorer computes three evidence dimensions:

| Dimension | Prefix | Main score column |
| --- | --- | --- |
| purpose_articulation | `pa` | `pa_sum_score` |
| history_consistency | `hc` | `hc_sum_score` |
| strategy_alignment | `sa` | `sa_sum_score` |

Each dimension combines lexical, TF-IDF, embedding, metadata, and prompt-pattern scores.
`history_consistency` also includes `hc_history_bonus_score`.

Expected outputs:

```text
data/phase2/evidence_score_v1_newdata.csv
data/phase2/evidence_score_v1_diagnostics_newdata.txt
```

## Phase 2B: Evidence Bucket Assignment

Run after Phase 2A finishes:

```powershell
python src\evidence_buckets_v1\run_evidence_buckets.py
```

Current config:

```text
src/evidence_buckets_v1/evidence_bucket_config.py
```

Input:

```text
data/phase2/evidence_score_v1_newdata.csv
```

Bucket thresholds:

| Bucket | Score column | Threshold |
| --- | --- | --- |
| purpose_articulation | `pa_sum_score` | `0.90` |
| history_consistency | `hc_sum_score` | `0.95` |
| strategy_alignment | `sa_sum_score` | `0.85` |

Per-bucket outputs:

```text
data/phase2/evidence_buckets_v1/purpose_articulation_evidence_v1_newdata.csv
data/phase2/evidence_buckets_v1/purpose_articulation_evidence_v1_newdata.db
data/phase2/evidence_buckets_v1/history_consistency_evidence_v1_newdata.csv
data/phase2/evidence_buckets_v1/history_consistency_evidence_v1_newdata.db
data/phase2/evidence_buckets_v1/strategy_alignment_evidence_v1_newdata.csv
data/phase2/evidence_buckets_v1/strategy_alignment_evidence_v1_newdata.db
```

Diagnostics and review-support outputs:

```text
data/phase2/evidence_buckets_v1/evidence_bucket_membership_v1_newdata.csv
data/phase2/evidence_buckets_v1/evidence_overlap_review_v1_newdata.csv
data/phase2/evidence_buckets_v1/evidence_overlap_diagnostics_v1_newdata.txt
```

The bucket step adds overlap and review fields such as:

```text
bucket_overlap_count
bucket_overlap_type
top_bucket_by_score
top_score
second_score
top_score_margin
needs_overlap_review
needs_margin_review
review_flag
```

## Phase 3: LLM Evidence Review

Run after Phase 2B finishes.

Small smoke test:

```powershell
python src\evidence_llm_review_v1\run_llm_evidence_review.py --bucket purpose_articulation --limit 5 --force
```

Full run:

```powershell
python src\evidence_llm_review_v1\run_llm_evidence_review.py --bucket all --resume
```

Useful arguments:

| Argument | Meaning |
| --- | --- |
| `--bucket all` | Process all evidence buckets. |
| `--bucket purpose_articulation` | Process one bucket only. Also supports `history_consistency` and `strategy_alignment`. |
| `--limit N` | Limit rows per bucket, useful for smoke tests. |
| `--force` | Overwrite existing output files. |
| `--resume` | Continue from existing outputs by skipping completed `chunk_id` values. |

Current config:

```text
src/evidence_llm_review_v1/llm_review_config.py
```

Inputs:

```text
data/phase2/evidence_buckets_v1/purpose_articulation_evidence_v1_newdata.csv
data/phase2/evidence_buckets_v1/history_consistency_evidence_v1_newdata.csv
data/phase2/evidence_buckets_v1/strategy_alignment_evidence_v1_newdata.csv
```

Outputs:

```text
data/phase3/evidence_llm_review_v1/purpose_articulation_llm_review_v1_newdata.csv
data/phase3/evidence_llm_review_v1/history_consistency_llm_review_v1_newdata.csv
data/phase3/evidence_llm_review_v1/strategy_alignment_llm_review_v1_newdata.csv
data/phase3/evidence_llm_review_v1/all_evidence_llm_reviews_v1_newdata.csv
data/phase3/evidence_llm_review_v1/human_review_queue_v1_newdata.csv
data/phase3/evidence_llm_review_v1/evidence_llm_review_diagnostics_v1_newdata.txt
```

Run metadata:

```text
RUN_VERSION = evidence_llm_review_v1_newdata
DEFAULT_LLM_MODEL = gpt-4o-mini
LLM_MODEL_ENV_VAR = OPENAI_MODEL
LLM_API_KEY_ENV_VAR = OPENAI_API_KEY
```

Set the API key before running Phase 3:

```powershell
$env:OPENAI_API_KEY="..."
```

Optionally override the model:

```powershell
$env:OPENAI_MODEL="gpt-4o-mini"
```

The script also tries to load `.env` through `python-dotenv` if installed, and can fall back to `configs.config.OPENAI_API_KEY` if present.

## Recommended Run Order

```powershell
python src\evidence_score_v1\pipeline\run_evidence_score.py
python src\evidence_buckets_v1\run_evidence_buckets.py
python src\evidence_llm_review_v1\run_llm_evidence_review.py --bucket purpose_articulation --limit 5 --force
python src\evidence_llm_review_v1\run_llm_evidence_review.py --bucket all --resume
```

Use the limited Phase 3 run first to verify API credentials, model responses, and output permissions before launching the full review.

## Quick Validation Commands

Compile the relevant scripts:

```powershell
python -m py_compile src\evidence_score_v1\pipeline\run_evidence_score.py src\evidence_buckets_v1\run_evidence_buckets.py src\evidence_llm_review_v1\run_llm_evidence_review.py
```

Check LLM review CLI imports without calling the API:

```powershell
python src\evidence_llm_review_v1\run_llm_evidence_review.py --help
```

Validate Phase 3 input paths and required columns without calling the API:

```powershell
python -c "import sys, pandas as pd; sys.path.insert(0, 'src'); import evidence_llm_review_v1.run_llm_evidence_review as r; [r.validate_input_df(pd.read_csv(r.input_path_for_bucket(b)), b) for b in r.EVIDENCE_POOL_CONFIG]; print('phase3 inputs OK')"
```

## Notes

- Run commands from the project root: `PAN_purpose0.1`.
- If a CSV is open in Excel or being synced by OneDrive, Phase 3 may fail the output writability check. Close the file or pause sync and rerun.
- On this Windows machine, PowerShell may print an execution-policy warning for the user profile. That warning does not necessarily mean the Python command failed; check the Python exit message and generated files.
- The current scripts are intended to support direct file execution, for example `python src\evidence_score_v1\pipeline\run_evidence_score.py`, not only `python -m ...`.
