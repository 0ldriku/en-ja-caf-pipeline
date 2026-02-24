# JA PauseVAD+Classifier (Neural Model) Run Log

## Model
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`

## Command 1: JA CAF with pause-local VAD + classifier
```powershell
python ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_pausevad_classifier.py `
  ja/rq3_v16_clean_release_20260223/auto_clauses `
  --audio-dir ja/results/manual20_0220_2 `
  --model-path shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt `
  --threshold 0.10 `
  --fallback-use-vad `
  --fallback-max-island-dur 0.25 `
  --output ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/auto_caf_results_pausevad_classifier_hybrid_best_mae_t010_fb025_neural.csv
```

## Command 2: Correlation probe
```powershell
python ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py `
  --baseline-auto-csv ja/rq3_v16_clean_release_20260223/auto_caf_results.csv `
  --vad-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv `
  --clf-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/auto_caf_results_pausevad_classifier_hybrid_best_mae_t010_fb025_neural.csv `
  --manual-csv ja/rq3_v16_clean_release_20260223/manual_caf_results.csv `
  --out-summary ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_summary_best_mae_t010_fb025_neural.csv `
  --out-mcpd-deltas ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_mcpd_file_deltas_best_mae_t010_fb025_neural.csv
```

## MCPD (baseline -> prior VAD -> neural classifier)
- Overall: `r 0.955 -> 0.952 -> 0.948`, `ICC 0.878 -> 0.892 -> 0.892`, `MAE 0.152 -> 0.137 -> 0.131`
- ST1: `r 0.960 -> 0.957 -> 0.951`, `ICC 0.881 -> 0.883 -> 0.889`, `MAE 0.130 -> 0.125 -> 0.114`
- ST2: `r 0.953 -> 0.950 -> 0.946`, `ICC 0.881 -> 0.901 -> 0.896`, `MAE 0.174 -> 0.148 -> 0.148`

## Outputs
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/auto_caf_results_pausevad_classifier_hybrid_best_mae_t010_fb025_neural.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_summary_best_mae_t010_fb025_neural.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_mcpd_file_deltas_best_mae_t010_fb025_neural.csv`
