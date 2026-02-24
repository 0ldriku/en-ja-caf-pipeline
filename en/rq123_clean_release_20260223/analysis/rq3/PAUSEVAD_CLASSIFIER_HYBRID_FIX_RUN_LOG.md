# Pause-VAD Classifier Hybrid Fix Run Log (Gold-40)

## What was wrong
- Previous classifier path scored ASR-gap candidates and was too conservative out-of-domain.
- Most VAD islands got very low filler probabilities, so recall was too low and MCPD stayed weak.

## Fix implemented
- New script: `en/rq123_clean_release_20260223/scripts/caf_calculator_pausevad_classifier.py`
- Core changes:
  - Run VAD **inside TextGrid pause intervals** (same units used by CAF).
  - Score each VAD island with classifier probability.
  - New hybrid fallback mode: if classifier predicts no island in a VAD-valid pause, allow short VAD islands (`--fallback-max-island-dur`) to preserve recall.

## Best configuration (from grid search)
- `--threshold 0.30`
- `--fallback-use-vad`
- `--fallback-max-island-dur 0.35`

## Command (best run)
```powershell
python en/rq123_clean_release_20260223/scripts/caf_calculator_pausevad_classifier.py `
  en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses `
  --audio-dir en/data/allsstar_full_manual/wav `
  --model-path shared/filler_classifier/model_podcastfillers_supervised_v1_smoke/model.joblib `
  --threshold 0.30 `
  --fallback-use-vad `
  --fallback-max-island-dur 0.35 `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected `
  --output en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_pausevad_classifier_hybrid_best.csv

python en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_classifier_probe.py `
  --auto-vad-csv en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv `
  --auto-clf-csv en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_pausevad_classifier_hybrid_best.csv `
  --out-dir en/rq123_clean_release_20260223/analysis/rq3/probe_pausevadclf_hybrid_best
```

## MCPD result (baseline -> old VAD -> hybrid)
- Overall: `r 0.816 -> 0.895 -> 0.903`, `ICC 0.788 -> 0.881 -> 0.892`, `MAE 0.093 -> 0.070 -> 0.063`
- ST1: `r 0.707 -> 0.842 -> 0.823`, `ICC 0.662 -> 0.832 -> 0.826`, `MAE 0.100 -> 0.070 -> 0.069`
- ST2: `r 0.908 -> 0.927 -> 0.947`, `ICC 0.878 -> 0.911 -> 0.929`, `MAE 0.086 -> 0.070 -> 0.057`

## Grid-search artifact
- `en/rq123_clean_release_20260223/analysis/rq3/pausevadclf_fallback_grid/fallback_grid_summary.csv`

## Output files
- `en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_pausevad_classifier_hybrid_best.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/probe_pausevadclf_hybrid_best/rq3_vad_classifier_probe_summary.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/probe_pausevadclf_hybrid_best/rq3_vad_classifier_probe_mcpd_file_deltas.csv`
