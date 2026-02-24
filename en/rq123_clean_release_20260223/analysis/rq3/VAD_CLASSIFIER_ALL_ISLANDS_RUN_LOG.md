# VAD+Classifier All-Islands Run Log (Gold-40)

## Date
- 2026-02-24

## Goal
- Change postprocess from single primary VAD island per ASR gap to scoring all VAD islands in the gap.
- Re-run CAF and RQ3 correlation probe on Gold-40.

## Script Patched
- `en/postprocess_vad_filler_classifier_en.py`
  - Removed single-island selection behavior.
  - New behavior: for VAD-kept gaps, classify every detected voiced island and emit one row per island.

## Commands
```powershell
python en/postprocess_vad_filler_classifier_en.py `
  --file-list-json en/annotation/selected_files.json `
  --file-list-key all_selected `
  --audio-dir en/data/allsstar_full_manual/wav `
  --asr-json-dir en/results/qwen3_filler_mfa_beam100/json `
  --model-path shared/filler_classifier/model_podcastfillers_supervised_v1_smoke/model.joblib `
  --out-dir en/analysis/vad_filler_postprocess_gold40_allislands

python en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py `
  en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses `
  --candidate-dir en/analysis/vad_filler_postprocess_gold40_allislands/per_file `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected `
  --output en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_classifier_allislands.csv

python en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_classifier_probe.py `
  --auto-vad-csv en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv `
  --auto-clf-csv en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_classifier_allislands.csv `
  --out-dir en/rq123_clean_release_20260223/analysis/rq3/probe_clf_allislands
```

## Postprocess Summary
- Files: 40/40 success
- Total candidate rows: 2039 (old single-primary: 1932)
- Scored rows: 1990 (old: 1883)
- Predicted fillers: 99 (same as old)

## Correlation (MCPD)
- Overall: baseline `r=.816 / ICC=.788 / MAE=.093` -> old VAD `r=.895 / ICC=.881 / MAE=.070` -> all-islands clf `r=.814 / ICC=.794 / MAE=.093`
- ST1: baseline `r=.707 / ICC=.662 / MAE=.100` -> old VAD `r=.842 / ICC=.832 / MAE=.070` -> all-islands clf `r=.702 / ICC=.668 / MAE=.101`
- ST2: baseline `r=.908 / ICC=.878 / MAE=.086` -> old VAD `r=.927 / ICC=.911 / MAE=.070` -> all-islands clf `r=.911 / ICC=.885 / MAE=.085`

## Diff Checks
- CAF CSV diff: `auto_caf_gold40_vad_classifier_allislands.csv` vs prior `auto_caf_gold40_vad_classifier.csv` -> no row-level metric changes across all 40 files.
- Multi-island diagnostics:
  - Gaps with >1 voiced island: 99
  - Gaps with non-primary positive island: 0
  - Gaps where primary was negative but another island was positive: 0

## Output Paths
- `en/analysis/vad_filler_postprocess_gold40_allislands/`
- `en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_classifier_allislands.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/probe_clf_allislands/rq3_vad_classifier_probe_summary.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/probe_clf_allislands/rq3_vad_classifier_probe_mcpd_file_deltas.csv`
