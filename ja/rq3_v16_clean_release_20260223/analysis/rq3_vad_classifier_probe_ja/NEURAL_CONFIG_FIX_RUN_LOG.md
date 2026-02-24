# JA Neural Config Fix Run Log

## Issue
Previous JA neural probe used fallback-heavy settings (`threshold=0.10` + `--fallback-use-vad`), which degraded ranking behavior.

## Fixed command
```powershell
python ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_pausevad_classifier.py `
  ja/rq3_v16_clean_release_20260223/auto_clauses `
  --audio-dir ja/results/manual20_0220_2 `
  --model-path shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt `
  --threshold 0.30 `
  --fallback-max-island-dur 0.25 `
  --output ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/auto_caf_results_pausevad_classifier_t030_nofallback_neural.csv
```

Note: `--fallback-use-vad` is intentionally NOT passed (fallback disabled).

## Probe command
```powershell
python ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py `
  --baseline-auto-csv ja/rq3_v16_clean_release_20260223/auto_caf_results.csv `
  --vad-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv `
  --clf-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/auto_caf_results_pausevad_classifier_t030_nofallback_neural.csv `
  --manual-csv ja/rq3_v16_clean_release_20260223/manual_caf_results.csv `
  --out-summary ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_summary_t030_nofallback_neural.csv `
  --out-mcpd-deltas ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_mcpd_file_deltas_t030_nofallback_neural.csv
```

## MCPD (baseline -> prior VAD -> fixed neural)
- Overall: `r 0.955 -> 0.952 -> 0.957`, `ICC 0.878 -> 0.892 -> 0.884`, `MAE 0.152 -> 0.137 -> 0.147`
- ST1: `r 0.960 -> 0.957 -> 0.959`, `ICC 0.881 -> 0.883 -> 0.883`, `MAE 0.130 -> 0.125 -> 0.126`
- ST2: `r 0.953 -> 0.950 -> 0.956`, `ICC 0.881 -> 0.901 -> 0.888`, `MAE 0.174 -> 0.148 -> 0.168`

## Outputs
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/auto_caf_results_pausevad_classifier_t030_nofallback_neural.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_summary_t030_nofallback_neural.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_mcpd_file_deltas_t030_nofallback_neural.csv`
