# RUN_LOG (Fresh-run canonical)

## Date
- 2026-02-24

## Scope
- This log tracks the final EN release pipeline used for reporting in this bundle.
- Legacy/test folders were removed; report inputs now point only to fresh-run outputs.

## Environment
- ASR and analysis runtime: `C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe`
- MFA runtime: `C:\Users\riku\miniconda3\envs\mfa\python.exe`
- Full dependency notes: `ENVIRONMENT.md`

## Final RQ3 pipeline used
1. ASR word-level TextGrid generation (already available in release input set)
- Script: `en/rq123_clean_release_20260223/asr_qwen3_mfa_en.py`
- Input audio: `en/data/allsstar_full_manual/wav`
- Output used: `en/results/qwen3_filler_mfa_beam100/textgrids_clean/` and `en/results/qwen3_filler_mfa_beam100/json/`

2. Manual-span blanking
- Not used for EN.

3. Clause segmentation (already available in release input set)
- Script: `en/rq123_clean_release_20260223/scripts/textgrid_caf_segmenter_v3.py`
- Output used:
  - Auto clauses: `en/results/qwen3_filler_mfa_beam100/clauses/`
  - Manual clauses: `en/results/manual_260212/clauses/`

4. Gap-only neural filler candidate scoring
```powershell
python en/postprocess_vad_filler_classifier_en.py `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected `
  --audio-dir en/data/allsstar_full_manual/wav `
  --asr-json-dir en/results/qwen3_filler_mfa_beam100/json `
  --model-path shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt `
  --threshold 0.50 `
  --gap-only `
  --out-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates
```

5. Auto CAF with candidate fillers
```powershell
python en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py `
  en/results/qwen3_filler_mfa_beam100/clauses `
  --candidate-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/per_file `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected `
  --output en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gold40_gaponly_neural_t050.csv
```

6. Manual CAF reference
- Script: `en/rq123_clean_release_20260223/scripts/caf_calculator.py`
- Output used: `en/results/manual_260212/caf_results_manual.csv`

7. Correlation summary used by the report
- Source summary CSV:
  - `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary_quality39.csv`
- Quality filter applied:
  - Excluded `ALL_139_M_PBR_ENG_ST1`
  - Final cohort: `n=39` (`ST1=19`, `ST2=20`)
- Final taskwise exports:
  - `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_overall.csv`
  - `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_st1.csv`
  - `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_st2.csv`

## Report files tied to this run
- `en/rq123_clean_release_20260223/RQ1_RQ2_REPORT.md`
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/RUN_LOG.md`
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/DATA_PROVENANCE.md`
