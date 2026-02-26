# Fresh-run Validation Report

Date: 2026-02-24
Folder: `release_en_ja_shared_20260224/_validation_freshrun_temp_20260224`

## Goal
Validate that the cleaned release structure still reproduces the same EN/JA final correlation outputs from the fresh-run pipeline stage (filler candidate scoring -> CAF -> correlation summary).

## Models used
- Filler classifier model used in rerun:
  - `release_en_ja_shared_20260224/shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`
- Disfluency model path check (one-file JA clause segmentation smoke run):
  - Loaded from `release_en_ja_shared_20260224/shared/disfluency_detector/model_v2/final`

## EN rerun outputs
- Candidates: `en/candidates/`
- Auto CAF: `en/auto_caf_gold40_gaponly_neural_t050.csv`
- Recomputed quality39 summary: `en/rq3_gaponly_neural_t050_probe_summary_quality39.csv`

Comparison against canonical:
- Canonical: `release_en_ja_shared_20260224/en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary_quality39.csv`
- Result: exact match.
- Max absolute diff = 0 for all key fields:
  - `n, mean_auto, mean_manual, pearson_r, spearman_rho, icc_2_1, mae`

## JA rerun outputs
- Candidates: `ja/candidates/`
- Auto CAF: `ja/auto_caf_gaponly_neural_t050.csv`
- Recomputed summary: `ja/rq3_gaponly_neural_t050_probe_summary.csv`

Comparison against canonical (`track=vad_classifier` rows):
- Canonical: `release_en_ja_shared_20260224/ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`
- Result: exact match.
- Max absolute diff = 0 for all key fields:
  - `n, mean_auto, mean_manual, pearson_r, spearman_rho, icc_2_1, mae`

## Conclusion
- Cleanup changes did not change EN/JA final fresh-run correlation results.
- Shared model usage is correct:
  - filler model: shared model path above
  - disfluency model: loaded from release shared path above
