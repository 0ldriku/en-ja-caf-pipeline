# EN PauseVAD+Classifier (Neural Model) Run Log

## Model
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`

## Command 1: CAF with pause-local VAD + classifier
```powershell
python en/rq123_clean_release_20260223/scripts/caf_calculator_pausevad_classifier.py `
  en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses `
  --audio-dir en/data/allsstar_full_manual/wav `
  --model-path shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt `
  --threshold 0.30 `
  --fallback-use-vad `
  --fallback-max-island-dur 0.35 `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected `
  --output en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_pausevad_classifier_hybrid_best_neural.csv
```

## Command 2: Correlation probe
```powershell
python en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_classifier_probe.py `
  --auto-vad-csv en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv `
  --auto-clf-csv en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_pausevad_classifier_hybrid_best_neural.csv `
  --out-dir en/rq123_clean_release_20260223/analysis/rq3/probe_pausevadclf_hybrid_best_neural
```

## MCPD (baseline -> prior VAD -> neural classifier)
- Overall: `r 0.816 -> 0.895 -> 0.906`, `ICC 0.788 -> 0.881 -> 0.898`, `MAE 0.093 -> 0.070 -> 0.062`
- ST1: `r 0.707 -> 0.842 -> 0.840`, `ICC 0.662 -> 0.832 -> 0.842`, `MAE 0.100 -> 0.070 -> 0.070`
- ST2: `r 0.908 -> 0.927 -> 0.950`, `ICC 0.878 -> 0.911 -> 0.932`, `MAE 0.086 -> 0.070 -> 0.053`

## Outputs
- `en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_pausevad_classifier_hybrid_best_neural.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/probe_pausevadclf_hybrid_best_neural/rq3_vad_classifier_probe_summary.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/probe_pausevadclf_hybrid_best_neural/rq3_vad_classifier_probe_mcpd_file_deltas.csv`
