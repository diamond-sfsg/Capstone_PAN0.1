# Evidence Pipeline Phases 2-4: Purpose-Driven Company Scoring

This document describes the complete evidence pipeline from Phase 2 evidence scoring through Phase 4 company purpose scoring.

## Overview

The pipeline processes company filings and disclosures to score companies on their "purpose-driven" behavior across three dimensions:
- **Purpose Articulation**: Clear expression of mission beyond profit
- **History Consistency**: Sustained commitment over time
- **Strategy Alignment**: Purpose embedded in business strategy and operations

```text
Phase 2A: Evidence Scoring
  Input:  data/clean_2.0/unified_chunks_v4.csv
  Script: src/evidence_score_v1/pipeline/run_evidence_score.py
  Output: data/phase2/evidence_score_v1_newdata.csv

Phase 2B: Evidence Bucket Assignment
  Input:  data/phase2/evidence_score_v1_newdata.csv
  Script: src/evidence_buckets_v1/run_evidence_buckets.py
  Output: data/phase2/evidence_buckets_v1/*_evidence_v1_newdata.csv

Phase 3: LLM Evidence Review
  Input:  data/phase2/evidence_buckets_v1/*_evidence_v1_newdata.csv
  Script: src/evidence_llm_review_v1/run_llm_evidence_review.py
  Output: data/phase3/evidence_llm_review_v1/all_evidence_llm_reviews_v1_newdata.csv

Phase 4: Company Purpose Scoring
  Input:  data/phase3/evidence_llm_review_v1/all_evidence_llm_reviews_v1_newdata.csv
  Script: src/company_purpose_v1/run_company_purpose_score.py
  Output: data/phase4/company_purpose_score_v1/*_newdata.csv
```

## Phase 2A: Evidence Scoring

**Purpose**: Compute relevance scores for each text chunk across the three purpose dimensions using multiple scoring methods.

**Process**:
1. Load unified chunks from `data/clean_2.0/unified_chunks_v4.csv`
2. For each dimension (PA/HC/SA), calculate scores using:
   - Lexical matching (keyword presence)
   - TF-IDF similarity
   - Embedding similarity (semantic)
   - Metadata matching (section context)
   - Prompt-pattern matching (LLM-based)
3. Combine scores into dimension-specific sums (e.g., `pa_sum_score`)
4. Output scored evidence matrix

**Key Files**:
- `src/evidence_score_v1/config.py`: Dimension definitions and scoring weights
- `src/evidence_score_v1/dimension_registry.py`: Dimension configurations
- `src/evidence_score_v1/pipeline/run_evidence_score.py`: Main scoring script

## Phase 2B: Evidence Bucket Assignment

**Purpose**: Categorize each evidence chunk into the most relevant purpose dimension bucket.

**Process**:
1. Load Phase 2A scored evidence
2. For each chunk, compare dimension scores against thresholds:
   - PA: `pa_sum_score >= 0.90`
   - HC: `hc_sum_score >= 0.95`
   - SA: `sa_sum_score >= 0.85`
3. Assign to highest-scoring dimension that meets threshold
4. Handle overlaps and edge cases with review flags
5. Output bucketed evidence files per dimension

**Key Files**:
- `src/evidence_buckets_v1/evidence_bucket_config.py`: Bucket thresholds and output paths
- `src/evidence_buckets_v1/run_evidence_buckets.py`: Bucket assignment logic

## Phase 3: LLM Evidence Review

**Purpose**: Use LLM to review and score evidence quality within each bucket, filtering out low-quality chunks.

**Process**:
1. Load bucketed evidence from Phase 2B
2. For each chunk in each bucket, prompt LLM to evaluate:
   - Relevance to assigned dimension
   - Evidence specificity and credibility
   - Source context quality
   - Boilerplate risk
   - Overall confidence
3. Generate additional scores like `llm_bucket_relevance_score`, `llm_credibility_score`
4. Flag chunks needing human review
5. Output consolidated reviewed evidence

**Key Files**:
- `src/evidence_llm_review_v1/llm_review_config.py`: LLM prompts and field definitions
- `src/evidence_llm_review_v1/prompt_templates.py`: Dimension-specific review prompts
- `src/evidence_llm_review_v1/run_llm_evidence_review.py`: LLM review orchestration

## Phase 4: Company Purpose Scoring

**Purpose**: Aggregate evidence per company-year and use LLM to generate final purpose scores.

**Process**:
1. Load Phase 3 reviewed evidence
2. For each company-year target:
   - Filter evidence by company and time window
   - Rank evidence within each dimension using RAG-style retrieval
   - Select top 8 most relevant chunks per dimension
   - Build evidence pack for LLM scoring
3. Prompt LLM to score company across three dimensions using rubric
4. Generate final 0-100 purpose score and labels
5. Output company-year scores and aggregated company scores

**Key Files**:
- `src/company_purpose_v1/purpose_score_config.py`: Scoring parameters and weights
- `src/company_purpose_v1/rubric_config.py`: Scoring rubric and dimension queries
- `src/company_purpose_v1/prompt_templates.py`: LLM scoring prompts
- `src/company_purpose_v1/run_company_purpose_score.py`: Main scoring orchestration

## Data Flow Summary

```
Raw Data → Phase 2A Scoring → Phase 2B Bucketing → Phase 3 LLM Review → Phase 4 Aggregation
```

Each phase builds on the previous, progressively filtering and enriching evidence to produce reliable company purpose scores.

## Key Concepts

- **Evidence Chunk**: Text segment from company filings with metadata
- **Dimension**: One of PA/HC/SA scoring aspects
- **Bucket**: Evidence assigned to its primary dimension
- **RAG Retrieval**: TF-IDF + quality scoring for evidence selection
- **Rubric Scoring**: Structured LLM evaluation against defined criteria

## Running the Pipeline

See individual phase READMEs for detailed run commands and requirements.