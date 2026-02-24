# VAD + Filler Classifier Postprocess Run Log

## Inputs
- file_list_json: `ja\rq3_v16_clean_release_20260223\analysis\rq3_gapclf_probe_ja\file_list_40.json`
- file_list_key: `all_selected`
- audio_dir: `ja\results\manual20_0220_2`
- asr_json_dir: `ja\results\qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20\json`
- model_path: `shared\filler_classifier\model_podcastfillers_neural_v1_full\model.pt`
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
- per_file CSV dir: `ja\rq3_v16_clean_release_20260223\analysis\rq3_gaponly_neural_t050_freshrun_20260224\candidates\per_file`
- summary by file: `ja\rq3_v16_clean_release_20260223\analysis\rq3_gaponly_neural_t050_freshrun_20260224\candidates\SUMMARY_BY_FILE.csv`
- all candidates: `ja\rq3_v16_clean_release_20260223\analysis\rq3_gaponly_neural_t050_freshrun_20260224\candidates\ALL_CANDIDATES.csv`
- run summary: `ja\rq3_v16_clean_release_20260223\analysis\rq3_gaponly_neural_t050_freshrun_20260224\candidates\RUN_SUMMARY.json`
- patched textgrids: (disabled)

## Aggregates
- requested files: 40
- success files: 40
- failed files: 0
- total gap candidates: 1724
- total scored: 1724
- total predicted fillers: 44
