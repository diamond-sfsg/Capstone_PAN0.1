# Purpose Articulation Code Flow

This module scores company-level Purpose Articulation (PA) from pooled-year evidence chunks.

## Entry Point

Run the pipeline with:

```powershell
python src\purpose_articulation\run_pa_score.py --input data\test\unified_chunks_test_v4.csv --output-dir data\test\pa_openai_output --provider openai
```

The runner loads chunks, builds company targets, retrieves evidence for each PA question, calls the selected LLM provider, aggregates question scores, and exports CSV/JSONL outputs.

## Main Files

- `run_pa_score.py`: command-line orchestration.
- `pa_config.py`: paths, retrieval parameters, scoring weights, source priors, keyword lists, and LLM settings.
- `pa_loader.py`: loads the chunk CSV, normalizes column names, source labels, text fields, years, and chunk IDs.
- `pa_rubric.py`: defines the three PA questions and score rubrics.
- `pa_retrieval.py`: ranks candidate evidence using keyword relevance, context completeness, source prior, and sparse similarity.
- `pa_evidence_score.py`: computes rule-based evidence quality signals and Q3 evidence-set quality.
- `pa_prompt_builder.py`: builds JSON-only prompts for evidence-level and evidence-set LLM scoring.
- `pa_llm_runner.py`: handles `mock`, `openai`, and `gemini` providers and normalizes LLM JSON responses.
- `pa_overlap.py`: reduces repeated contribution when the same chunk supports multiple PA questions.
- `pa_aggregator.py`: converts evidence-level and evidence-set scores into question and company scores.
- `pa_year_status.py` / `pa_year_stats.py`: produce year/source coverage diagnostics.
- `pa_exporter.py`: writes company, question, evidence-detail, raw LLM, and diagnostics outputs.

## Scoring Flow

1. `load_chunks()` reads the input CSV and standardizes it to the PA schema.
2. `build_company_targets()` produces pooled-year company targets.
3. For each company, `filter_company()` selects all available evidence chunks.
4. Q1 and Q2 use evidence-level scoring:
   - `retrieve_candidates_for_question()` ranks candidate chunks.
   - `select_llm_evidence_q1_q2()` keeps the top evidence rows for LLM scoring.
  - `PAEvaluator.score_evidence()` scores each evidence row and retains an `extracted_purpose` sentence when present.
   - `apply_overlap_to_evidence_rows()` applies reuse penalties.
   - `aggregate_evidence_question_score()` combines top evidence into a question score.
5. Q3 uses evidence-set scoring:
   - `build_q3_evidence_set()` selects source-balanced evidence.
   - `compute_q3_evidence_set_quality()` summarizes source diversity and formal/strategic evidence.
  - `PAEvaluator.score_evidence_set()` scores the set and retains an evidence-set-level `extracted_purpose` sentence when present.
   - `aggregate_q3_score()` applies the evidence-set quality factor.
6. `aggregate_company_pa_score()` averages Q1, Q2, and Q3 into the company PA score.
7. `compute_year_stats()` and `compute_source_mix()` add diagnostics.
8. `export_all()` writes final outputs.

## Outputs

The pipeline writes:

- `pa_company_score_v1.csv`: company-level PA scores and diagnostics.
- `pa_question_score_v1.csv`: Q1/Q2/Q3 scores per company.
- `pa_evidence_detail_v1.csv`: selected evidence rows, scoring details, and LLM-extracted purpose text for evidence retention.
- `pa_evidence_library_v1.csv`: retrieval-stage evidence returned for scoring, including Q1/Q2 candidate pools, Q1/Q2 LLM-selected evidence, and Q3 evidence-set selections.
- `pa_llm_raw_outputs_v1.jsonl`: raw and parsed LLM outputs for audit, including `extracted_purpose`.
- `pa_diagnostics_v1.txt`: run summary and coverage diagnostics.

## Providers

- `mock`: deterministic local scoring for smoke tests.
- `openai`: OpenAI API scoring using `OPENAI_API_KEY` and `OPENAI_MODEL`.
- `gemini`: Gemini scoring using `GEMINI_API_KEY` and `GEMINI_MODEL`.

Use `mock` to validate plumbing. Use `openai` or `gemini` for rubric-based scoring.
