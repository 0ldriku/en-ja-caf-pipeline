# JA VAD + Classifier Probe Run Log

## Date
- 2026-02-24

## Goal
Test pause-local VAD + classifier refinement on JA RQ3 (40 files) and compare against:
1. baseline (`auto_caf_results.csv`)
2. prior JA VAD tuned probe (`analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv`)

## New Scripts
- `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_pausevad_classifier.py`
- `ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py`

## Inputs
- Auto clauses: `ja/rq3_v16_clean_release_20260223/auto_clauses`
- Manual CAF: `ja/rq3_v16_clean_release_20260223/manual_caf_results.csv`
- Audio dir: `ja/results/manual20_0220_2` (40 wav)
- Classifier model: `shared/filler_classifier/model_podcastfillers_only_v1/model.joblib`

## Commands

### Single test (EN-like config)
```powershell
python ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_pausevad_classifier.py `
  ja/rq3_v16_clean_release_20260223/auto_clauses `
  --audio-dir ja/results/manual20_0220_2 `
  --model-path shared/filler_classifier/model_podcastfillers_only_v1/model.joblib `
  --threshold 0.30 `
  --fallback-use-vad `
  --fallback-max-island-dur 0.35 `
  --output ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/auto_caf_results_pausevad_classifier_hybrid_t030_fb035.csv

python ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py `
  --baseline-auto-csv ja/rq3_v16_clean_release_20260223/auto_caf_results.csv `
  --vad-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv `
  --clf-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/auto_caf_results_pausevad_classifier_hybrid_t030_fb035.csv `
  --manual-csv ja/rq3_v16_clean_release_20260223/manual_caf_results.csv `
  --out-summary ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_summary.csv `
  --out-mcpd-deltas ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja/rq3_vad_classifier_probe_mcpd_file_deltas.csv
```

### Sweep
- thresholds: `0.10, 0.20, 0.30, 0.40, 0.50`
- fallback max island duration: `0.25, 0.35, 0.45, 0.55`
- sweep summary: `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja_sweep/sweep_summary.csv`

## Key Sweep Findings (MCPD)

### Best by Pearson r (Overall)
- Config: `threshold=0.20`, `fallback_max_island_dur=0.25`
- Overall: `r=0.949`, `ICC=0.894`, `MAE=0.128`
- ST1: `r=0.952`, `ICC=0.894`, `MAE=0.111`
- ST2: `r=0.947`, `ICC=0.897`, `MAE=0.144`

### Best by MAE / highest ICC (Overall)
- Config: `threshold=0.10`, `fallback_max_island_dur=0.25`
- Overall: `r=0.946`, `ICC=0.898`, `MAE=0.121`
- ST1: `r=0.949`, `ICC=0.900`, `MAE=0.103`
- ST2: `r=0.945`, `ICC=0.899`, `MAE=0.139`

## Baseline/VAD reference (MCPD)
- Baseline: Overall `r=0.955`, `ICC=0.878`, `MAE=0.152`
- Prior VAD: Overall `r=0.952`, `ICC=0.892`, `MAE=0.137`

Interpretation:
- Hybrid probe consistently lowered MCPD MAE and raised ICC vs baseline and prior VAD.
- Pearson r decreased vs baseline/prior VAD in this JA set.

## Artifacts
- Sweep: `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_classifier_probe_ja_sweep/`
- Best-by-MAE files:
  - `auto_caf_results_pausevad_classifier_hybrid_best_mae_t010_fb025.csv`
  - `rq3_vad_classifier_probe_summary_best_mae_t010_fb025.csv`
  - `rq3_vad_classifier_probe_mcpd_file_deltas_best_mae_t010_fb025.csv`
- Best-by-r files:
  - `auto_caf_results_pausevad_classifier_hybrid_best_r_t020_fb025.csv`
  - `rq3_vad_classifier_probe_summary_best_r_t020_fb025.csv`
  - `rq3_vad_classifier_probe_mcpd_file_deltas_best_r_t020_fb025.csv`
