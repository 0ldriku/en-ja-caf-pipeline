# VAD Probe Run Log (Gold-40)

## Date
- 2026-02-23

## Goal
- Test whether audio-based pause refinement can reduce MCPD drift caused by ASR-missed filler tokens.

## Scripts
- `en/rq123_clean_release_20260223/scripts/caf_calculator_vad.py`
- `en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_probe.py`

## Inputs
- Auto clause TextGrids:
  - `en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses`
- Manual CAF reference CSV:
  - `en/rq123_clean_release_20260223/results/manual_260212/caf_results_manual.csv`
- Gold file list:
  - `en/rq123_clean_release_20260223/annotation/selected_files.json` (`all_selected`)
- Audio:
  - `en/data/allsstar_full_manual/wav`

## Command (tuned setting)
```powershell
$env:PYTHONIOENCODING='utf-8'

& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" `
  "en/rq123_clean_release_20260223/scripts/caf_calculator_vad.py" `
  "en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses" `
  --use-audio-vad `
  --audio-dir "en/data/allsstar_full_manual/wav" `
  --file-list-json "en/rq123_clean_release_20260223/annotation/selected_files.json" `
  --file-list-key all_selected `
  --vad-top-db 30 `
  --vad-min-occupancy 0.2 `
  --vad-min-voiced 0.15 `
  --vad-merge-gap 0.1 `
  --output "en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv"

& "C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe" `
  "en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_probe.py" `
  --auto-vad-csv "en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv" `
  --out-dir "en/rq123_clean_release_20260223/analysis/rq3"
```

## VAD refinement summary (tuned)
- `vad_pause_candidates`: 1902
- `vad_pauses_split`: 235
- `vad_speech_islands`: 309

## Main result (MCPD, baseline -> VAD tuned)
- Overall: `r .816 -> .895`, `ICC .788 -> .881`, `MAE .093 -> .070`
- ST1: `r .707 -> .842`, `ICC .662 -> .832`, `MAE .100 -> .070`
- ST2: `r .908 -> .927`, `ICC .878 -> .911`, `MAE .086 -> .070`

## Outputs
- `en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/rq3_vad_probe_summary.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/rq3_vad_probe_mcpd_file_deltas.csv`

## Notes
- Root mechanism confirmed on `ALL_034_M_HIN_ENG_ST1`: ASR misses filler tokens and merges hesitation gaps into a long mid-clause pause; VAD split reduces this inflation.
- Tradeoff remains: AR and ECPD agreement drop slightly in this VAD mode because pause structure is modified globally.
