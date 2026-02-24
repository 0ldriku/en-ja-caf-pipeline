# Analysis — English CAF Pipeline Validation

**Last updated:** 2025-02-14

## Overview

Three research questions evaluated using edit-distance alignment (Matsuura et al., 2025):

| RQ | Question | Script | Output |
|:--|:--|:--|:--|
| RQ1 | Clause boundary agreement (auto vs gold) | `rq1/run_rq1_gold.py` | `rq1/rq1_clause_boundary_gold.csv` |
| RQ2 | Pause location agreement (MCP/ECP, auto vs gold) | `rq2/run_rq2_gold.py` | `rq2/rq2_pause_location_gold.csv` |
| RQ3 | Concurrent validity of 9 CAF measures (auto vs manual) | `rq3/run_rq3_validity.py` | `rq3/rq3_concurrent_validity.csv` |

## Usage

```bash
cd en/analysis
.venv/bin/python rq1/run_rq1_gold.py
.venv/bin/python rq2/run_rq2_gold.py
.venv/bin/python rq3/run_rq3_validity.py
```

## Data sources

| Data | Path | Files |
|:--|:--|--:|
| Auto clause TextGrids | `results/qwen3_filler_mfa_beam100/clauses/` | 190 |
| Auto CAF results | `results/qwen3_filler_mfa_beam100/caf_results_beam100.csv` | 190 |
| Manual CAF results | `results/manual_260212/caf_results_manual.csv` | 190 |
| Gold clause boundaries (adjudicated) | `annotation/boundary_agreement_260213/final_correct_segments/` | 10 |
| Gold clause boundaries (LLM production) | `annotation/llm_output/production_30/` | 30 |
| Canonical transcripts | `annotation/transcripts/` | 40 |
| Selected files list | `annotation/selected_files.json` | 40 |

## Results summary (2025-02-14)

### RQ1: Clause boundary agreement (n=40)

| Metric | Overall | ST1 | ST2 |
|:--|--:|--:|--:|
| F1 (micro) | **.845** | .869 | .826 |
| κ (micro) | **.816** | .845 | .795 |

### RQ2: Pause location agreement (n=40)

| Metric | Overall | ST1 | ST2 |
|:--|--:|--:|--:|
| κ | **.840** | .873 | .815 |
| Accuracy | .921 | .937 | .909 |

### RQ3: Concurrent validity of CAF measures (n=174, 16 excluded)

| Category | Measures | Pearson *r* | ICC(2,1) |
|:--|:--|--:|--:|
| Speed | AR, SR | .942–.985 | .938–.980 |
| Pause rate | MCPR, ECPR, PR | .893–.961 | .887–.946 |
| Composite | MLR | .962 | .939 |
| Pause duration | MCPD, ECPD, MPD | .888–.954 | .863–.950 |

All correlations *p* < .001.

## Reports and documentation

| Document | Description |
|:--|:--|
| `RQ1_RQ2_REPORT.md` | Full RQ1–RQ3 results report with tables and interpretation |
| `PIPELINE_OVERVIEW.md` | Pipeline architecture, script details, annotation methodology |

## Legacy scripts (superseded)

The following scripts in this folder are from earlier pipeline versions and are **no longer used**:

| Script | Replaced by |
|:--|:--|
| `run_correlation.py` | `rq3/run_rq3_validity.py` |
| `run_pause_agreement.py` | `rq2/run_rq2_gold.py` |
| `run_clause_agreement.py` | `rq1/run_rq1_gold.py` |
| `show_excluded_files.py` | Exclusions hardcoded in `rq3/run_rq3_validity.py` |
