# EN RQ1-RQ3 Clean Release

This folder is a clean package of the English analysis pipeline assets used for RQ1-RQ3.

## Folder

- `en/rq123_clean_release_20260223`

## Included Scripts

- `asr_qwen3_mfa_en.py`
- `scripts/textgrid_caf_segmenter_v3.py`
- `scripts/caf_calculator.py`
- `../../shared/filler_classifier/candidate_review_textgrids.py` (candidate CSV <-> review TextGrid bridge for manual revision)
- `../../shared/filler_classifier/train_podcastfillers_neural_classifier.py` (neural filler model training script)
- `analysis/rq1/run_rq1_gold.py`
- `analysis/rq2/run_rq2_gold.py`
- `analysis/rq3/run_rq3_validity.py`

## Included Inputs

- `results/qwen3_filler_mfa_beam100/textgrids_clean/` (190)
- `results/qwen3_filler_mfa_beam100/clauses/` (190 TextGrids + clause log)
- `results/qwen3_filler_mfa_beam100/caf_results_beam100.csv`
- `results/manual_260212/clauses/` (190 TextGrids + clause log)
- `results/manual_260212/caf_results_manual.csv`
- `annotation/selected_files.json`
- `annotation/transcripts/`
- `annotation/boundary_agreement_260213/final_correct_segments/`
- `annotation/llm_output/production_30/`

## Included Outputs

- `analysis/rq1/rq1_clause_boundary_gold.csv`
- `analysis/rq2/rq2_pause_location_gold.csv`
- `analysis/rq3/rq3_concurrent_validity.csv`
- `analysis/rq3/rq3_file_level.csv`

## Included Documentation

- `docs/PIPELINE_OVERVIEW.md`
- `docs/ANALYSIS_README.md`
- `RQ1_RQ2_REPORT.md`
- `ENVIRONMENT.md`
- `results/qwen3_filler_mfa_beam100/RUN_LOG.md`
- `results/manual_260212/RUN_LOG.md`

## Environment

- Exact conda/venv + dependency notes are in:
  - `en/rq123_clean_release_20260223/ENVIRONMENT.md`

## Re-run (from repo root)

Use UTF-8 stdout on Windows to avoid console encoding errors from Greek/special symbols in script prints.

```powershell
$env:PYTHONIOENCODING='utf-8'
& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" "en/rq123_clean_release_20260223/analysis/rq1/run_rq1_gold.py"
& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" "en/rq123_clean_release_20260223/analysis/rq2/run_rq2_gold.py"
& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" "en/rq123_clean_release_20260223/analysis/rq3/run_rq3_validity.py"
```

## Notes

- RQ scripts resolve paths relative to this release root, so they run without editing paths.
- The disfluency model is not duplicated in this folder; scripts auto-discover:
  - `shared/disfluency_detector/model_v2/final` (preferred canonical path)
  - `en/disfluency_test/l2_disfluency_detector/model_v2/final` (legacy fallback)
- Filler model used by EN/JA gap-only fresh reruns: `../../shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`.
- Neural model training/reference details: `../../shared/filler_classifier/README.md`.
- Gap/classifier candidate manual-review workflow:
  - `analysis/rq3_gaponly_neural_t050_20260224/review_textgrids/`
  - `analysis/rq3_gaponly_neural_t050_20260224/candidates_from_review_tier_t050/`
  - `analysis/rq3_gaponly_neural_t050_20260224/REVIEW_WORKFLOW.md`
