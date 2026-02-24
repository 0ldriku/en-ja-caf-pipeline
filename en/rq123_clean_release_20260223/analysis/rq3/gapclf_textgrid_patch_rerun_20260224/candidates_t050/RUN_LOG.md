# VAD + Filler Classifier Postprocess Run Log

## Inputs
- file_list_json: `en\annotation\selected_files.json`
- file_list_key: `all_selected`
- audio_dir: `en\data\allsstar_full_manual\wav`
- asr_json_dir: `en\results\qwen3_filler_mfa_beam100\json`
- model_path: `shared\filler_classifier\model_podcastfillers_only_v1\model.joblib`

## Params
- min_gap: 0.12
- max_gap: 1.2
- vad_top_db: 30.0
- vad_min_voiced: 0.08
- vad_merge_gap: 0.06
- min_occupancy: 0.2
- max_occupancy: 1.01
- threshold: 0.5

## Outputs
- per_file CSV dir: `en\rq123_clean_release_20260223\analysis\rq3\gapclf_textgrid_patch_rerun_20260224\candidates_t050\per_file`
- summary by file: `en\rq123_clean_release_20260223\analysis\rq3\gapclf_textgrid_patch_rerun_20260224\candidates_t050\SUMMARY_BY_FILE.csv`
- all candidates: `en\rq123_clean_release_20260223\analysis\rq3\gapclf_textgrid_patch_rerun_20260224\candidates_t050\ALL_CANDIDATES.csv`
- run summary: `en\rq123_clean_release_20260223\analysis\rq3\gapclf_textgrid_patch_rerun_20260224\candidates_t050\RUN_SUMMARY.json`
- patched textgrids: `en\rq123_clean_release_20260223\analysis\rq3\gapclf_textgrid_patch_rerun_20260224\patched_textgrids_t050`

## Aggregates
- requested files: 40
- success files: 40
- failed files: 0
- total gap candidates: 2039
- total scored: 1990
- total predicted fillers: 31
