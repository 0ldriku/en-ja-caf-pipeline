# JA Fresh-Run Data Provenance

This file records exactly which data folders were used for:
- filler candidate extraction,
- CAF recomputation,
- and correlation probe.

## Run folder
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/`

## Inputs

### File list
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json` (`all_selected`, 40 files)

### Audio used by filler detector
- `ja/results/manual20_0220_2`

### ASR JSON used by filler detector
- `ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/json`

### Clause TextGrids used by CAF recomputation
- `ja/rq3_v16_clean_release_20260223/auto_clauses`

### Manual CAF reference for probe
- `ja/rq3_v16_clean_release_20260223/manual_caf_results.csv`

### Baseline and prior-VAD CAF references for probe
- baseline: `ja/rq3_v16_clean_release_20260223/auto_caf_results.csv`
- prior VAD: `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv`

### Neural filler model
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`

## Fresh-run outputs
- candidates: `.../candidates/`
- auto CAF: `.../auto_caf_gaponly_neural_t050.csv`
- canonical probe summary: `.../probe/rq3_gaponly_neural_t050_probe_summary.csv`
- canonical probe deltas: `.../probe/rq3_gaponly_neural_t050_probe_mcpd_file_deltas.csv`
