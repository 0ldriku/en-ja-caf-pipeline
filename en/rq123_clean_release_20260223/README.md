# EN RQ1-RQ3 Clean Release (Fresh-run only)

This folder contains the English release artifacts used in the final fresh-run reporting.

## Scope

- RQ1/RQ2 gold evaluation outputs
- RQ3 final fresh-run outputs (quality-filtered Gold-39)
- Scripts used for ASR, clause segmentation, CAF, and correlation

## Canonical RQ3 outputs

- Fresh-run folder:
  - `analysis/rq3_gaponly_neural_t050_freshrun_20260224/`
- Final source summary (quality-filtered):
  - `analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary_quality39.csv`
- Final taskwise exports:
  - `../../analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_overall.csv`
  - `../../analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_st1.csv`
  - `../../analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_st2.csv`

## Scripts used

- `asr_qwen3_mfa_en.py`
- `scripts/textgrid_caf_segmenter_v3.py`
- `scripts/caf_calculator.py`
- `scripts/caf_calculator_vad_classifier.py`
- `scripts/run_rq3_vad_classifier_probe_en.py`
- `../../postprocess_vad_filler_classifier_en.py`

## Reports

- `RQ1_RQ2_REPORT.md`
- `RUN_LOG.md`
- `ENVIRONMENT.md`

## Notes

- EN RQ3 final cohort is quality-filtered Gold-39 (`ST1=19`, `ST2=20`), excluding `ALL_139_M_PBR_ENG_ST1`.
- Legacy test/probe folders were removed from this release bundle.
