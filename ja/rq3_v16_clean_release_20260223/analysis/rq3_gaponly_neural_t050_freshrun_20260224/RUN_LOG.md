# JA RQ3 Fresh Run Log (Gap-only neural)

Date: 2026-02-24

## Commands used

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

python ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py `
  ja/rq3_v16_clean_release_20260223/auto_clauses `
  --candidate-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/per_file `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/file_list_40.json `
  --file-list-key all_selected `
  --output ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gaponly_neural_t050.csv

python ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py `
  --baseline-auto-csv ja/rq3_v16_clean_release_20260223/auto_caf_results.csv `
  --clf-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gaponly_neural_t050.csv `
  --manual-csv ja/rq3_v16_clean_release_20260223/manual_caf_results.csv `
  --out-summary ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv `
  --out-mcpd-deltas ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_mcpd_file_deltas.csv
```

## Final reporting basis

- Source summary used for JA report:  
  `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`
- Final taskwise tables exported to:  
  `analysis_final_taskwise_correlations_20260224/ja/`

## Retained outputs in this folder

- `file_list_40.json`
- `candidates/`
- `auto_caf_gaponly_neural_t050.csv`
- `probe/rq3_gaponly_neural_t050_probe_summary.csv`
- `probe/rq3_gaponly_neural_t050_probe_mcpd_file_deltas.csv`
