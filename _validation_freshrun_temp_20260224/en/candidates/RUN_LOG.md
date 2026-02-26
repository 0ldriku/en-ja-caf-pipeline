# VAD + Filler Classifier Postprocess Run Log

## Inputs
- file_list_json: `release_en_ja_shared_20260224\en\rq123_clean_release_20260223\annotation\selected_files.json`
- file_list_key: `all_selected`
- audio_dir: `en\data\allsstar_full_manual\wav`
- asr_json_dir: `en\results\qwen3_filler_mfa_beam100\json`
- model_path: `release_en_ja_shared_20260224\shared\filler_classifier\model_podcastfillers_neural_v1_full\model.pt`
- model_kind: `neural`

## Params
- min_gap: 0.12
- max_gap: 1.2
- gap_only: True
- vad_top_db: 30.0
- vad_min_voiced: 0.08
- vad_merge_gap: 0.06
- min_occupancy: 0.2
- max_occupancy: 1.01
- threshold: 0.5

## Outputs
- per_file CSV dir: `release_en_ja_shared_20260224\_validation_freshrun_temp_20260224\en\candidates\per_file`
- summary by file: `release_en_ja_shared_20260224\_validation_freshrun_temp_20260224\en\candidates\SUMMARY_BY_FILE.csv`
- all candidates: `release_en_ja_shared_20260224\_validation_freshrun_temp_20260224\en\candidates\ALL_CANDIDATES.csv`
- run summary: `release_en_ja_shared_20260224\_validation_freshrun_temp_20260224\en\candidates\RUN_SUMMARY.json`
- patched textgrids: (disabled)

## Aggregates
- requested files: 40
- success files: 40
- failed files: 0
- total gap candidates: 1932
- total scored: 1932
- total predicted fillers: 42
