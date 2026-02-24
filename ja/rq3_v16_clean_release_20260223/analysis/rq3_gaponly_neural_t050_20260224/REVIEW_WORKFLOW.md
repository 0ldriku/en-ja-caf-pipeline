# JA Candidate Review Workflow (TextGrid + CSV)

This workflow lets a coder review neural gap candidates in Praat and rerun JA CAF/correlation from edited intervals.

## 1) Build review TextGrids

```powershell
python shared/filler_classifier/candidate_review_textgrids.py build `
  --candidate-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/candidates/per_file `
  --source-textgrid-dir ja/rq3_v16_clean_release_20260223/inputs/asr_word_textgrids `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json `
  --file-list-key all_selected `
  --audio-dir ja/results/manual20_0220_2 `
  --out-textgrid-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/review_textgrids `
  --out-audio-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/review_textgrids/audio
```

Output:
- `review_textgrids/*.TextGrid`
- `review_textgrids/audio/*.wav`
- `review_textgrids/REVIEW_BUILD_SUMMARY.csv`

## 2) Edit in Praat

Per file, tiers are:
- `words_original`: source ASR words tier
- `cand_all`: all candidate spans with metadata (`gap/island/prob/pred/duration`)
- `cand_pred_auto`: auto-predicted filler candidates only
- `review_edit`: editable tier used for final decision
- `words_patched_auto`: auto-patched words tier (`<filler_speech>` inserted)

Coder action:
- Edit `review_edit` only (delete/move/split/add intervals).
- Keep blanks where no filler insertion should occur.

## 3) Extract edited candidates back to CSV

```powershell
python shared/filler_classifier/candidate_review_textgrids.py extract `
  --review-textgrid-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/review_textgrids `
  --out-candidate-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/candidates_from_review_tier_t050 `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json `
  --file-list-key all_selected
```

Output:
- `candidates_from_review_tier_t050/*_vad_classifier.csv`
- `candidates_from_review_tier_t050/REVIEW_EXTRACT_SUMMARY.csv`

## 4) Recompute CAF and probe from reviewed candidates

```powershell
python ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py `
  ja/rq3_v16_clean_release_20260223/auto_clauses `
  --candidate-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/candidates_from_review_tier_t050 `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json `
  --file-list-key all_selected `
  --output ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/auto_caf_gaponly_neural_t050_reviewed.csv

python ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py `
  --baseline-auto-csv ja/rq3_v16_clean_release_20260223/auto_caf_results.csv `
  --vad-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv `
  --clf-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/auto_caf_gaponly_neural_t050_reviewed.csv `
  --manual-csv ja/rq3_v16_clean_release_20260223/manual_caf_results.csv `
  --out-summary ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/rq3_gaponly_neural_t050_summary_reviewed.csv `
  --out-mcpd-deltas ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/rq3_gaponly_neural_t050_mcpd_file_deltas_reviewed.csv
```

