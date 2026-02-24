# JA Gap-Only Neural Fresh Rerun (from filler detection)

Date: 2026-02-24

## Scope
- Fresh rerun starting from filler detection (candidate extraction), then CAF recomputation, then JA probe.
- No reuse of prior candidate CSVs in this folder.

## Commands

```powershell
python en/postprocess_vad_filler_classifier_en.py `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json `
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
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json `
  --file-list-key all_selected `
  --output ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gaponly_neural_t050.csv

python ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py `
  --baseline-auto-csv ja/rq3_v16_clean_release_20260223/auto_caf_results.csv `
  --vad-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv `
  --clf-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gaponly_neural_t050.csv `
  --manual-csv ja/rq3_v16_clean_release_20260223/manual_caf_results.csv `
  --out-summary ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/rq3_gaponly_neural_t050_summary.csv `
  --out-mcpd-deltas ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/rq3_gaponly_neural_t050_mcpd_file_deltas.csv
```

## Candidate Summary
- requested/success: `40/40`
- total gap candidates: `1724`
- total scored: `1724`
- total predicted fillers: `44`

## MCPD (baseline -> prior VAD -> fresh gap-only neural)
- Overall: `r 0.955 -> 0.952 -> 0.955`, `ICC 0.878 -> 0.892 -> 0.878`, `MAE 0.152 -> 0.137 -> 0.151`
- ST1: `r 0.960 -> 0.957 -> 0.959`, `ICC 0.881 -> 0.883 -> 0.877`, `MAE 0.130 -> 0.125 -> 0.130`
- ST2: `r 0.953 -> 0.950 -> 0.953`, `ICC 0.881 -> 0.901 -> 0.882`, `MAE 0.174 -> 0.148 -> 0.172`

## Outputs
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gaponly_neural_t050.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv` (canonical)
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_mcpd_file_deltas.csv` (canonical)
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/DATA_PROVENANCE.md`

Compatibility aliases (same content):
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/rq3_gaponly_neural_t050_summary.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/rq3_gaponly_neural_t050_mcpd_file_deltas.csv`
