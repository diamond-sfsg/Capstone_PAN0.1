# API Key And Data Setup

## API Keys

Put local API keys in:

```text
configs/config.py
```

Use this format:

```python
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

`configs/config.py` is already ignored by `.gitignore`, so it is intended for local secrets only.

You can also use environment variables instead:

```bat
set OPENAI_API_KEY=your-openai-api-key
set ANTHROPIC_API_KEY=your-anthropic-api-key
set CLAUDE_MODEL=claude-opus-4-1-20250805
```

Do not commit real API keys. If a real key was copied into chat, logs, or Git history, rotate it in the provider dashboard.

## Input Data

Raw source data should go under:

```text
data/raw/
```

Cleaned production input should go under:

```text
data/clean_2.0/
```

The main pipeline input file is expected here by default:

```text
data/clean_2.0/unified_chunks_final_v4.csv
```

Required columns include:

```text
chunk_id, doc_id, company, year, source, source_file, section, subsection,
text_raw, text_clean, token_count, char_count
```

## Test Data

Generated test datasets live under:

```text
data/test/
```

Current test dataset layout:

```text
data/test/test1_rand/unified_chunks_final_v4.csv
data/test/test2_num/unified_chunks_final_v4.csv
data/test/test3_year/unified_chunks_final_v4.csv
data/test/testc50/unified_chunks_final_v4.csv
```

Each test folder also has:

```text
selected_companies.csv
```

## Running Aggregate With Claude

Use:

```bat
run_aggregate_claude.bat data\test\test1_rand\unified_chunks_final_v4.csv outputs\aggregate_test1_rand_claude outout\aggregate\test1_rand_claude
```

Arguments:

```text
1. input CSV
2. phase output root
3. aggregate output directory
4. optional Claude model id
```

Example with explicit model:

```bat
run_aggregate_claude.bat data\clean_2.0\unified_chunks_final_v4.csv outputs\aggregate_claude outout\aggregate\claude claude-opus-4-1-20250805
```

The run is resumable. Progress is tracked at:

```text
<phase_output_root>/progress_company_phase.json
```

Progress keys use:

```text
COMPANY_PHASE
```

Examples:

```text
CHRW_PA
CHRW_SA
CHRW_HC
```

## Outputs

Per-company phase outputs are written under:

```text
<phase_output_root>/company_runs/<company>/pa/
<phase_output_root>/company_runs/<company>/sa/
<phase_output_root>/company_runs/<company>/hc/
```

Merged phase outputs are written under:

```text
<phase_output_root>/pa/
<phase_output_root>/sa/
<phase_output_root>/hc/
```

Final aggregate outputs are written under the aggregate output directory:

```text
Summary.csv
aggregate_company_summary.csv
aggregate_rubric_components.csv
aggregate_detail.csv
aggregate_run_progress.csv
```

`Summary.csv` is the main company-level result table. It contains PA, SA, HC, and aggregate mean scores.
