# VAD + Classifier Correlation Probe Run Log (Gold-40)

## Goal
Compare RQ3 concurrent-validity correlations for three auto tracks on the same Gold-40 files:
1. Baseline (`caf_results_beam100.csv`)
2. Previous audio-VAD refinement (`auto_caf_gold40_vad_tuned.csv`)
3. New classifier-guided pause refinement (`auto_caf_gold40_vad_classifier.csv`)

## Scripts Used
- `en/postprocess_vad_filler_classifier_en.py`
- `en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py`
- `en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_classifier_probe.py`

## Commands
```powershell
python "en/postprocess_vad_filler_classifier_en.py" `
  --file-list-json "en/annotation/selected_files.json" `
  --file-list-key "all_selected" `
  --audio-dir "en/data/allsstar_full_manual/wav" `
  --asr-json-dir "en/results/qwen3_filler_mfa_beam100/json" `
  --model-path "shared/filler_classifier/model_podcastfillers_supervised_v1_smoke/model.joblib" `
  --out-dir "en/analysis/vad_filler_postprocess_gold40"

python "en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py" `
  "en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/clauses" `
  --candidate-dir "en/analysis/vad_filler_postprocess_gold40/per_file" `
  --file-list-json "en/rq123_clean_release_20260223/annotation/selected_files.json" `
  --file-list-key "all_selected" `
  --output "en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_classifier.csv"

python "en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_classifier_probe.py" `
  --auto-vad-csv "en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_tuned.csv" `
  --auto-clf-csv "en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_classifier.csv" `
  --out-dir "en/rq123_clean_release_20260223/analysis/rq3"
```

## Key Intermediate Counts
From classifier postprocess (`en/analysis/vad_filler_postprocess_gold40/RUN_SUMMARY.json`):
- Requested files: 40
- Success files: 40
- Total gap candidates: 1932
- Scored by classifier: 1883
- Predicted fillers: 99

From CAF classifier refinement (`auto_caf_gold40_vad_classifier.csv`):
- Pause candidates: 1902
- Pauses removed/split by classifier predictions: 99
- Refined pauses: 1803

## Main MCPD Result (Pearson / ICC / MAE)
- Overall: baseline `0.816 / 0.788 / 0.093` -> old VAD `0.895 / 0.881 / 0.070` -> new clf `0.814 / 0.794 / 0.093`
- ST1: baseline `0.707 / 0.662 / 0.100` -> old VAD `0.842 / 0.832 / 0.070` -> new clf `0.702 / 0.668 / 0.101`
- ST2: baseline `0.908 / 0.878 / 0.086` -> old VAD `0.927 / 0.911 / 0.070` -> new clf `0.911 / 0.885 / 0.085`

## Output Files
- `en/rq123_clean_release_20260223/analysis/rq3/auto_caf_gold40_vad_classifier.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/rq3_vad_classifier_probe_summary.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/rq3_vad_classifier_probe_mcpd_file_deltas.csv`
- `en/rq123_clean_release_20260223/analysis/rq3/rq3_vad_classifier_probe_overall_table.csv`
