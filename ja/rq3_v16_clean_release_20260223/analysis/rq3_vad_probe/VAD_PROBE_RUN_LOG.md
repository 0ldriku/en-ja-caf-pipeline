# JA VAD Probe Run Log (v16 clean release)

Date: 2026-02-23

## Inputs

- Auto clauses: `ja/rq3_v16_clean_release_20260223/auto_clauses`
- Baseline auto CAF: `ja/rq3_v16_clean_release_20260223/auto_caf_results.csv`
- Manual CAF: `ja/rq3_v16_clean_release_20260223/manual_caf_results.csv`
- Audio dir: `ja/data/dataset_l1_focus20/audio` (mp3)

## Scripts

- `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_vad.py`
- `ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_probe_ja.py`

## Commands

```powershell
python "ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_vad.py" `
  "ja/rq3_v16_clean_release_20260223/auto_clauses" `
  --use-audio-vad `
  --audio-dir "ja/data/dataset_l1_focus20/audio" `
  --vad-top-db 30 `
  --vad-min-occupancy 0.2 `
  --vad-min-voiced 0.15 `
  --vad-merge-gap 0.1 `
  --output "ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv"

python "ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_probe_ja.py" `
  --baseline-auto-csv "ja/rq3_v16_clean_release_20260223/auto_caf_results.csv" `
  --vad-auto-csv "ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv" `
  --manual-csv "ja/rq3_v16_clean_release_20260223/manual_caf_results.csv" `
  --out-summary "ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/rq3_vad_probe_summary.csv" `
  --out-mcpd-deltas "ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/rq3_vad_probe_mcpd_file_deltas.csv"
```

## VAD Refinement Summary

- Files matched: 40
- `vad_pause_candidates`: 1860
- `vad_pauses_split`: 48
- `vad_speech_islands`: 164

## MCPD (baseline -> VAD)

- Overall: `r 0.955 -> 0.952`, `ICC 0.878 -> 0.892`, `MAE 0.152 -> 0.137`
- ST1: `r 0.960 -> 0.957`, `ICC 0.881 -> 0.883`, `MAE 0.130 -> 0.125`
- ST2: `r 0.953 -> 0.950`, `ICC 0.881 -> 0.901`, `MAE 0.174 -> 0.148`

## Notes

- VAD helps MCPD error on a small subset (8 improved, 1 worse, 31 unchanged).
- Tradeoff exists: several global duration/rate measures become worse than baseline (see `rq3_vad_probe_summary.csv`).
