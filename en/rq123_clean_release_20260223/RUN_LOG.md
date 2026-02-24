# RUN_LOG

## Date
- 2026-02-23

## Goal
Create a single clean English package (scripts + data + outputs) for RQ1-RQ3, similar to the Japanese clean package.

## Clean Package
- `en/rq123_clean_release_20260223`

## Environment and Dependencies

- Full pinned environment notes:
  - `en/rq123_clean_release_20260223/ENVIRONMENT.md`
- Primary runtime for EN analysis scripts:
  - `C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe` (Python 3.12.12)
- MFA runtime used by ASR stage:
  - `C:\Users\riku\miniconda3\envs\mfa\python.exe` (Python 3.10.19)

## Sources Copied

### Scripts
- `en/asr_qwen3_mfa_en.py`
- `en/scripts/textgrid_caf_segmenter_v3.py`
- `en/scripts/caf_calculator.py`
- `en/analysis/rq1/run_rq1_gold.py`
- `en/analysis/rq2/run_rq2_gold.py`
- `en/analysis/rq3/run_rq3_validity.py`

### Inputs
- `en/results/qwen3_filler_mfa_beam100/textgrids_clean/`
- `en/results/qwen3_filler_mfa_beam100/clauses/`
- `en/results/qwen3_filler_mfa_beam100/caf_results_beam100.csv`
- `en/results/manual_260212/clauses/`
- `en/results/manual_260212/caf_results_manual.csv`
- `en/annotation/selected_files.json`
- `en/annotation/transcripts/`
- `en/annotation/boundary_agreement_260213/final_correct_segments/`
- `en/annotation/llm_output/production_30/`

### Existing Outputs + Docs
- `en/analysis/rq1/rq1_clause_boundary_gold.csv`
- `en/analysis/rq2/rq2_pause_location_gold.csv`
- `en/analysis/rq3/rq3_concurrent_validity.csv`
- `en/analysis/rq3/rq3_file_level.csv`
- `en/analysis/PIPELINE_OVERVIEW.md`
- `en/analysis/README.md`
- `en/analysis/RQ1_RQ2_REPORT.md`
- `en/results/qwen3_filler_mfa_beam100/RUN_LOG.md`
- `en/results/manual_260212/RUN_LOG.md`

## Validation Commands Executed

```powershell
$env:PYTHONIOENCODING='utf-8'
& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" "en/rq123_clean_release_20260223/analysis/rq1/run_rq1_gold.py"
& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" "en/rq123_clean_release_20260223/analysis/rq2/run_rq2_gold.py"
& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" "en/rq123_clean_release_20260223/analysis/rq3/run_rq3_validity.py"
& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" "en/rq123_clean_release_20260223/analysis/rq3/run_rq3_validity_dualtrack.py"
```

## Validation Result
- All 4 scripts completed successfully in the release folder.
- RQ1 summary reproduced: micro F1=0.845, kappa=0.816.
- RQ2 summary reproduced: kappa=0.840, accuracy=0.921.
- RQ3 summary reproduced: Pearson r range 0.888-0.985, ICC range 0.863-0.980.
- RQ3 dual-track summary produced:
  - `full_quality_174`: Pearson r range 0.888-0.985, ICC range 0.863-0.980
  - `gold40_raw`: Pearson r range 0.816-0.987, ICC range 0.788-0.985
  - `gold40_quality_39`: Pearson r range 0.820-0.988, ICC range 0.796-0.984

## Data Integrity Check
- Compared release outputs vs original outputs:
  - `rq1_clause_boundary_gold.csv`: numerically identical.
  - `rq2_pause_location_gold.csv`: numerically identical.
  - `rq3_file_level.csv`: numerically identical.
  - `rq3_concurrent_validity.csv`: numerically identical except negligible floating-point serialization difference in one p-value field.
  - New dual-track outputs generated in release folder:
    - `analysis/rq3/rq3_dualtrack_summary.csv`
    - `analysis/rq3/rq3_dualtrack_file_membership.csv`

## Counts in Release
- `results/qwen3_filler_mfa_beam100/textgrids_clean`: 190 files
- `results/qwen3_filler_mfa_beam100/clauses`: 191 files (190 TextGrids + log)
- `results/manual_260212/clauses`: 191 files (190 TextGrids + log)
- `annotation/transcripts`: 41 files
- `annotation/boundary_agreement_260213/final_correct_segments`: 10 files
- `annotation/llm_output/production_30`: 30 files
