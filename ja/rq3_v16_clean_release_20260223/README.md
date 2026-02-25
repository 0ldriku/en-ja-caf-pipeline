# JA RQ3 Clean Release (Fresh-run only)

This folder contains the Japanese RQ3 release artifacts used in the final fresh-run reporting.

## Scope

- Final manual gold reference (`manual_clauses_gold_v2`)
- Final fresh-run RQ3 outputs (40 files: ST1=20, ST2=20)
- Scripts used for clause segmentation, CAF, and correlation

## Canonical RQ3 outputs

- Fresh-run folder:
  - `analysis/rq3_gaponly_neural_t050_freshrun_20260224/`
- Final source summary:
  - `analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`
- Final taskwise exports:
  - `../../analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_overall.csv`
  - `../../analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_st1.csv`
  - `../../analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_st2.csv`

## Scripts used

- `scripts/ja_clause_segmenter_v16.py`
- `scripts/caf_calculator_ja.py`
- `scripts/caf_calculator_ja_gap_classifier.py`
- `scripts/run_rq3_vad_classifier_probe_ja.py`
- `../../en/postprocess_vad_filler_classifier_en.py`

## Reports

- `RQ3_VALIDITY_REPORT_JA.md`
- `RUN_LOG.md`
- `ENVIRONMENT.md`

## Notes

- `manual_clauses_gold_v2/` is the final gold standard used for JA reporting.
- Legacy test/probe folders were removed from this release bundle.
