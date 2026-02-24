# Candidate Review TextGrid Build Log

## Inputs
- candidate_dir: `en\rq123_clean_release_20260223\analysis\rq3_gaponly_neural_t050_20260224\candidates\per_file`
- source_textgrid_dir: `en\rq123_clean_release_20260223\results\qwen3_filler_mfa_beam100\textgrids_clean`
- file_list_json: `en\rq123_clean_release_20260223\annotation\selected_files.json`
- file_list_key: `all_selected`
- candidate_suffix: `_vad_classifier.csv`
- filler_label: `<filler_speech>`

## Outputs
- review_textgrid_dir: `en\rq123_clean_release_20260223\analysis\rq3_gaponly_neural_t050_20260224\review_textgrids`
- review_summary_csv: `en\rq123_clean_release_20260223\analysis\rq3_gaponly_neural_t050_20260224\review_textgrids\REVIEW_BUILD_SUMMARY.csv`
- copied_audio_dir: `en\rq123_clean_release_20260223\analysis\rq3_gaponly_neural_t050_20260224\review_textgrids\audio`

## Counts
- files_total: 40
- files_success: 40
- files_failed: 0
- total_candidates: 1932
- total_predicted: 42
