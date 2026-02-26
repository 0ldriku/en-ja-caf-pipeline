# JA RQ3 Fresh Run Data Provenance

Canonical fresh-run folder:
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/`

Inputs:
- ASR JSON: `ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/json/`
- ASR clause TextGrids: `ja/rq3_v16_clean_release_20260223/auto_clauses/`
- Manual CAF reference: `ja/rq3_v16_clean_release_20260223/manual_caf_results.csv`
- File list: `file_list_40.json` (local in this folder)

Model:
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`

Reporting outputs:
- Source summary: `probe/rq3_gaponly_neural_t050_probe_summary.csv`
- Final taskwise export bundle: `analysis_final_taskwise_correlations_20260224/ja/`
