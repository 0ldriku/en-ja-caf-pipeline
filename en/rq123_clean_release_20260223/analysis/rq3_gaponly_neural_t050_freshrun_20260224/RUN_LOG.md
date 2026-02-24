# EN Gap-Only Neural Fresh Rerun (from filler detection)

Date: 2026-02-24

## Scope
- Fresh rerun starting from filler detection (candidate extraction), then CAF recomputation, then correlation probe.
- No reuse of prior candidate CSVs in this folder.

## Commands

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

python en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_classifier_probe.py `
  --auto-vad-csv en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv `
  --auto-clf-csv en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gold40_gaponly_neural_t050.csv `
  --out-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe
```

## Candidate Summary
- requested/success: `40/40`
- total gap candidates: `1932`
- total scored: `1932`
- total predicted fillers: `42`

## MCPD (baseline -> prior VAD -> fresh gap-only neural)
- Overall: `r 0.816 -> 0.895 -> 0.817`, `ICC 0.788 -> 0.881 -> 0.790`, `MAE 0.093 -> 0.070 -> 0.092`
- ST1: `r 0.707 -> 0.842 -> 0.704`, `ICC 0.662 -> 0.832 -> 0.656`, `MAE 0.100 -> 0.070 -> 0.101`
- ST2: `r 0.908 -> 0.927 -> 0.910`, `ICC 0.878 -> 0.911 -> 0.886`, `MAE 0.086 -> 0.070 -> 0.084`

## Outputs
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/`
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gold40_gaponly_neural_t050.csv`
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv` (canonical)
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_mcpd_file_deltas.csv` (canonical)
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/DATA_PROVENANCE.md`

Compatibility aliases (same content):
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_vad_classifier_probe_summary.csv`
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_vad_classifier_probe_mcpd_file_deltas.csv`
