# JA RQ3 Clean Release (v16)

This folder is a clean package of the files used for the Japanese RQ3 validity run.

## Contents

- `scripts/`
  - `ja_clause_segmenter_v16.py` (release copy includes robust auto-discovery of disfluency model path)
  - `caf_calculator_ja.py`
  - `correlation_from_caf_ja.py`
  - `apply_gold_v2_fixes.py`
  - `../../shared/filler_classifier/candidate_review_textgrids.py` (candidate CSV <-> review TextGrid bridge for manual revision)
  - `../../shared/filler_classifier/train_podcastfillers_neural_classifier.py` (neural filler model training script)
  - `asr/`
    - `asr_qwen3_mfa_ja_v2_spanfix_test_b.py` (ASR+MFA script used for `spanfix_b` source TextGrids)
    - `asr_qwen3_mfa_ja_v2.py` (baseline ASR+MFA script)
    - `make_beginning_removed_by_manual.py` (post-process to blank ASR leading labels using manual onset)

- `inputs/asr_word_textgrids/`
  - ASR word-level TextGrids used as segmentation input.

- `inputs/manual_word_textgrids/`
  - Manual word-level TextGrids used as segmentation input.

- `auto_clauses/`
  - v16 clause-segmented ASR TextGrids.
- `manual_clauses/`
  - Manual clause TextGrids produced by v16.
- `manual_clauses_gold_v1/`
  - First-pass manually corrected gold clauses.
- `manual_clauses_gold_v2/`
  - Final manually corrected gold clauses used as reference.
  - Includes `clause_log.txt` when available.
- `GOLD_V2_FIX_REPORT.md`
  - Manual gold v2 fix notes.

- `auto_caf_results.csv`
- `manual_caf_results.csv`
- `rq3_concurrent_validity_ja.csv`
- `rq3_file_level_ja.csv`

- `RUN_LOG.md`
  - Reproducible command log for this package.
- `RQ3_VALIDITY_REPORT_JA.md`
  - Final report for this package.
- `ENVIRONMENT.md`
  - Exact conda/venv + dependency notes used for ASR and RQ3 stages.
- `analysis/rq3_gaponly_neural_t050_20260224/REVIEW_WORKFLOW.md`
  - Manual-review workflow from candidate CSVs to review TextGrids and back to CAF-ready candidate CSVs.
- Filler model used by EN/JA gap-only runs:
  - `../../shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`
- Neural model training/reference details:
  - `../../shared/filler_classifier/README.md`

## Expected Counts

- `inputs/asr_word_textgrids/*.TextGrid`: 40
- `inputs/manual_word_textgrids/*.TextGrid`: 40
- `auto_clauses/*.TextGrid`: 40
- `manual_clauses/*.TextGrid`: 40
- `manual_clauses_gold_v1/*.TextGrid`: 40
- `manual_clauses_gold_v2/*.TextGrid`: 40

## ASR Provenance

- ASR input used in this package (`inputs/asr_word_textgrids/*.TextGrid`) was sourced from:
  - `ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual`
- Upstream generation flow:
  1. Run `scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_test_b.py` on selected audio to produce `textgrids_clean`.
  2. Run `scripts/asr/make_beginning_removed_by_manual.py` with manual TextGrids to create leading-blanked ASR TextGrids in the same style as `textgrids_clean_beginning_removed_by_manual`.
