# V5 Verification Notes (Claim Discipline + Bias-Aware Framing)

This note records manuscript claims in `draft/paper_draftv5.tex` that were checked against code/results files.

## Architecture and Flow Checks

- Pipeline stage wording updated to match implemented flow.
  - Sources:
    - `README.md`
    - `en/rq123_clean_release_20260223/scripts/asr/asr_qwen3_mfa_en.py`
    - `ja/rq3_v16_clean_release_20260223/scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_b.py`
    - `shared/postprocess_vad_filler_classifier_en.py`
    - `en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py`

- Filler-augmented alignment constants confirmed and manuscript wording tightened.
  - Verified constants: `GAP_MIN=0.4`, `GAP_OFFSET=0.35`, `GAP_STEP=0.55`, `FILLER_MAX=3`, MFA `--beam 100 --retry_beam 400`
  - Sources:
    - `en/rq123_clean_release_20260223/scripts/asr/asr_qwen3_mfa_en.py`
    - `ja/rq3_v16_clean_release_20260223/scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_b.py`

- Sentence-segmentation description corrected (EN uses wtpsplit JSON; JA uses GiNZA sentence boundaries + rule-based re-splitting).
  - Sources:
    - `en/rq123_clean_release_20260223/scripts/textgrid_caf_segmenter_v3.py`
    - `ja/rq3_v16_clean_release_20260223/scripts/ja_clause_segmenter_v16.py`

- Pause MCP/ECP classification wording corrected to match code logic (150 ms onset-to-clause-end check; midpoint-in-clause check; otherwise ECP).
  - Sources:
    - `en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py`
    - `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py`

## Analysis and Result-Consistency Checks

- English RQ1/RQ2 headline values confirmed.
  - RQ1 micro: `F1=.845`, `kappa=.816`, boundaries `N=1131`, mean WER `=.121`
  - RQ2 overall: `kappa=.840`, accuracy `=.921`, pauses `N=1902`
  - Sources:
    - `en/rq123_clean_release_20260223/RQ1_RQ2_REPORT.md`
    - `en/rq123_clean_release_20260223/analysis/rq1/rq1_clause_boundary_gold.csv`
    - `en/rq123_clean_release_20260223/analysis/rq2/rq2_pause_location_gold.csv`

- Overstated file-level RQ1 wording corrected.
  - Previous statement implied both lowest-kappa files had high WER; actual pair includes one low-WER file (`0.083`).
  - Source:
    - `en/rq123_clean_release_20260223/analysis/rq1/rq1_clause_boundary_gold.csv`

- Overstated "lowest RQ1" linkage in RQ2 section corrected.
  - Lowest RQ2-accuracy file has `kappa=.578` in RQ1, but absolute minimum RQ1 kappa is `.567` in another file.
  - Sources:
    - `en/rq123_clean_release_20260223/analysis/rq1/rq1_clause_boundary_gold.csv`
    - `en/rq123_clean_release_20260223/analysis/rq2/rq2_pause_location_gold.csv`

- Discussion WER-stratified kappa numbers corrected.
  - WER `> .20`: mean kappa `.77`
  - WER `<= .20`: mean kappa `.83`
  - Source:
    - `en/rq123_clean_release_20260223/analysis/rq1/rq1_clause_boundary_gold.csv`

## Bias-Aware Statistics Framing Checks

- EN/JA RQ3 correlations and duration-bias direction retained and checked against released correlation bundles.
  - EN overall correlation bundle:
    - `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_all_subsets.csv`
    - `en/rq123_clean_release_20260223/RQ1_RQ2_REPORT.md`
  - JA overall correlation bundle:
    - `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_all_subsets.csv`
    - `ja/rq3_v16_clean_release_20260223/RQ3_VALIDITY_REPORT_JA.md`

- Filler model performance values in methods checked.
  - Validation F1 `0.9409` and test F1 `0.9328` (rounded to `.941` and `.933`).
  - Sources:
    - `shared/filler_classifier/model_podcastfillers_neural_v1_full/metrics.json`
    - `shared/filler_classifier/model_podcastfillers_neural_v1_full/README.md`
