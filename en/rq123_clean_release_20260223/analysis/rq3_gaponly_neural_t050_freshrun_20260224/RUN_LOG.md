# EN RQ3 Fresh Run Log (Gap-only neural, quality-filtered reporting)

Date: 2026-02-24

## Commands used

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

python en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py `
  en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses `
  --candidate-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/per_file `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected `
  --output en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gold40_gaponly_neural_t050.csv
```

## Final reporting basis

- Source summary used for EN report:  
  `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary_quality39.csv`
- Quality exclusion applied in reporting: `ALL_139_M_PBR_ENG_ST1`
- Final taskwise tables exported to:  
  `analysis_final_taskwise_correlations_20260224/en/`

## Retained outputs in this folder

- `candidates/`
- `auto_caf_gold40_gaponly_neural_t050.csv`
- `probe/rq3_gaponly_neural_t050_probe_summary_quality39.csv`
