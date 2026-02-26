# EN RQ3 Fresh Run Data Provenance

Canonical fresh-run folder:
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/`

Inputs:
- ASR JSON: `en/results/qwen3_filler_mfa_beam100/json/`
- ASR clause TextGrids: `en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses/`
- Manual CAF reference: `en/rq123_clean_release_20260223/results/manual_260212/caf_results_manual.csv`
- File list: `en/rq123_clean_release_20260223/annotation/selected_files.json` (`all_selected`)

Model:
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`

Reporting outputs:
- Source summary (quality-filtered): `probe/rq3_gaponly_neural_t050_probe_summary_quality39.csv`
- Final taskwise export bundle: `analysis_final_taskwise_correlations_20260224/en/`
