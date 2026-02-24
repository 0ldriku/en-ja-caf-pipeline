# EN Fresh-Run Data Provenance

This file records exactly which data folders were used for:
- filler candidate extraction,
- CAF recomputation,
- and correlation probe.

## Run folder
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/`

## Inputs

### File list
- `en/rq123_clean_release_20260223/annotation/selected_files.json` (`all_selected`, 40 files)

### Audio used by filler detector
- `en/data/allsstar_full_manual/wav`

### ASR JSON used by filler detector
- `en/results/qwen3_filler_mfa_beam100/json`

### Clause TextGrids used by CAF recomputation
- `en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses`

### Manual CAF reference for probe
- `en/results/manual_260212/caf_results_manual.csv`

### Baseline and prior-VAD CAF references for probe
- baseline: `en/results/qwen3_filler_mfa_beam100/caf_results_beam100.csv`
- prior VAD: `en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv`

### Neural filler model
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`

## Fresh-run outputs
- candidates: `.../candidates/`
- auto CAF: `.../auto_caf_gold40_gaponly_neural_t050.csv`
- canonical probe summary: `.../probe/rq3_gaponly_neural_t050_probe_summary.csv`
- canonical probe deltas: `.../probe/rq3_gaponly_neural_t050_probe_mcpd_file_deltas.csv`
