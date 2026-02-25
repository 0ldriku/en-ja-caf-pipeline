# RUN_LOG (Fresh-run canonical)

## Date
- 2026-02-24

## Scope
- This log tracks the final JA RQ3 release pipeline used for reporting in this bundle.
- Legacy/test folders were removed; report inputs now point only to fresh-run outputs.

## Environment
- Clause/CAF/correlation runtime: `ja/.venv_electra310/Scripts/python.exe`
- Upstream ASR runtime: `C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe`
- Upstream MFA runtime: `C:\Users\riku\miniconda3\envs\mfa\python.exe`
- Full dependency notes: `ENVIRONMENT.md`

## Final RQ3 pipeline used
1. ASR word-level TextGrid generation (already available in release input set)
- Script: `ja/rq3_v16_clean_release_20260223/scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_b.py`
- Output source used:
  - `ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean/`
  - `ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/json/`

2. Manual-span blanking (JA dataset-specific)
- Script: `ja/rq3_v16_clean_release_20260223/scripts/asr/make_beginning_removed_by_manual.py`
- Output source used:
  - `ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual/`
- Note: this is specific to this JA manual dataset coverage mismatch and is not a general requirement.

3. Clause segmentation
```powershell
ja/.venv_electra310/Scripts/python.exe ja/rq3_v16_clean_release_20260223/scripts/ja_clause_segmenter_v16.py `
  -i ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual `
  -o ja/rq3_v16_clean_release_20260223/auto_clauses `
  --model ja_ginza_electra

ja/.venv_electra310/Scripts/python.exe ja/rq3_v16_clean_release_20260223/scripts/ja_clause_segmenter_v16.py `
  -i ja/results/manual20_0220_2 `
  -o ja/rq3_v16_clean_release_20260223/manual_clauses `
  --model ja_ginza_electra
```

4. Manual gold reference
- Final gold used by report: `manual_clauses_gold_v2/`

5. Gap-only neural filler candidate scoring
```powershell
python en/postprocess_vad_filler_classifier_en.py `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/file_list_40.json `
  --file-list-key all_selected `
  --audio-dir ja/results/manual20_0220_2 `
  --asr-json-dir ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/json `
  --model-path shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt `
  --threshold 0.50 `
  --gap-only `
  --out-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates
```

6. Auto CAF with candidate fillers
```powershell
ja/.venv_electra310/Scripts/python.exe ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py `
  ja/rq3_v16_clean_release_20260223/auto_clauses `
  --candidate-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/per_file `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/file_list_40.json `
  --file-list-key all_selected `
  --output ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gaponly_neural_t050.csv
```

7. Manual CAF and correlation
- Manual CAF script: `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja.py`
- Probe/correlation script: `ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py`
- Source summary CSV used by report:
  - `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`
- Final taskwise exports:
  - `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_overall.csv`
  - `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_st1.csv`
  - `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_st2.csv`

## Final counts
- Matched files: `40` (`ST1=20`, `ST2=20`)

## Report files tied to this run
- `ja/rq3_v16_clean_release_20260223/RQ3_VALIDITY_REPORT_JA.md`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/RUN_LOG.md`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/DATA_PROVENANCE.md`
