# EN Candidate Review Workflow (TextGrid + CSV)

This workflow lets a coder review neural gap candidates in Praat and then rerun CAF/correlation from edited intervals.

## 1) Build review TextGrids

```powershell
python shared/filler_classifier/candidate_review_textgrids.py build `
  --candidate-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/candidates/per_file `
  --source-textgrid-dir en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/textgrids_clean `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected `
  --audio-dir en/data/allsstar_full_manual/wav `
  --out-textgrid-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/review_textgrids `
  --out-audio-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/review_textgrids/audio
```

Output:
- `review_textgrids/*.TextGrid`
- `review_textgrids/audio/*.wav`
- `review_textgrids/REVIEW_BUILD_SUMMARY.csv`

## 2) Edit in Praat

Per file, tiers are:
- `words_original`: source ASR words tier
- `cand_all`: every candidate with metadata label (`gap/island/prob/pred/duration`)
- `cand_pred_auto`: auto-predicted filler candidates only
- `review_edit`: editable tier used for final decision
- `words_patched_auto`: auto-patched words tier (`<filler_speech>` inserted)

Coder action:
- Edit only `review_edit` (delete, move, split, add intervals as needed).
- Leave blank where no filler should be inserted.

## 3) Extract edited candidates back to CSV

```powershell
python shared/filler_classifier/candidate_review_textgrids.py extract `
  --review-textgrid-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/review_textgrids `
  --out-candidate-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/candidates_from_review_tier_t050 `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected
```

Output:
- `candidates_from_review_tier_t050/*_vad_classifier.csv`
- `candidates_from_review_tier_t050/REVIEW_EXTRACT_SUMMARY.csv`

## 4) Recompute CAF and correlation from reviewed candidates

```powershell
python en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py `
  en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses `
  --candidate-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/candidates_from_review_tier_t050 `
  --file-list-json en/rq123_clean_release_20260223/annotation/selected_files.json `
  --file-list-key all_selected `
  --output en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/auto_caf_gold40_gaponly_neural_t050_reviewed.csv

python en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_classifier_probe.py `
  --auto-vad-csv en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv `
  --auto-clf-csv en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/auto_caf_gold40_gaponly_neural_t050_reviewed.csv `
  --out-dir en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/probe_reviewed
```

