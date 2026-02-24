# JA Gap-Classifier Probe (EN-Style)

## Goal
Run Japanese RQ3 comparison using the same classifier flow style as EN gap-candidate pipeline:
- candidate generation from ASR JSON gaps + VAD-like voiced islands + classifier
- CAF refinement from predicted candidate islands

## Inputs
- Auto clauses: `ja/rq3_v16_clean_release_20260223/auto_clauses`
- Manual CAF: `ja/rq3_v16_clean_release_20260223/manual_caf_results.csv`
- Baseline auto CAF: `ja/rq3_v16_clean_release_20260223/auto_caf_results.csv`
- Prior JA VAD CAF: `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv`
- Audio: `ja/results/manual20_0220_2`
- ASR JSON: `ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/json`
- Classifier model: `shared/filler_classifier/model_podcastfillers_only_v1/model.joblib`

## Commands

### 1) Build 40-file list
```powershell
@'
import json
from pathlib import Path
ids = sorted([p.stem for p in Path('ja/rq3_v16_clean_release_20260223/auto_clauses').glob('*.TextGrid')])
out = Path('ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'all_selected': ids}, ensure_ascii=False, indent=2), encoding='utf-8')
print('rewrote', out, 'count', len(ids))
'@ | python -
```

### 2) Generate gap candidates (EN script reused with JA paths)
```powershell
python en/postprocess_vad_filler_classifier_en.py `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json `
  --file-list-key all_selected `
  --audio-dir ja/results/manual20_0220_2 `
  --asr-json-dir ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/json `
  --model-path shared/filler_classifier/model_podcastfillers_only_v1/model.joblib `
  --out-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/candidates
```

Summary:
- requested/success: 40/40
- total gap candidates: 1767
- total scored: 1750
- total predicted fillers: 43
- threshold: 0.5

### 3) Compute JA CAF with gap-candidate classifier
```powershell
python ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py `
  ja/rq3_v16_clean_release_20260223/auto_clauses `
  --candidate-dir ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/candidates/per_file `
  --file-list-json ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/file_list_40.json `
  --file-list-key all_selected `
  --output ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/auto_caf_gap_classifier_t050.csv
```

Stats:
- `clf_pause_candidates`: 1860
- `clf_pauses_split`: 39
- `clf_predicted_speech_islands`: 39

### 4) Compare baseline vs VAD vs gap-classifier
```powershell
python ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py `
  --baseline-auto-csv ja/rq3_v16_clean_release_20260223/auto_caf_results.csv `
  --vad-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv `
  --clf-auto-csv ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/auto_caf_gap_classifier_t050.csv `
  --manual-csv ja/rq3_v16_clean_release_20260223/manual_caf_results.csv `
  --out-summary ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/rq3_gapclf_probe_summary.csv `
  --out-mcpd-deltas ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/rq3_gapclf_probe_mcpd_file_deltas.csv
```

Printed MCPD summary:
- Overall: r `0.955 -> 0.952 -> 0.955`, ICC `0.878 -> 0.892 -> 0.881`, MAE `0.152 -> 0.137 -> 0.147`
- ST1: r `0.960 -> 0.957 -> 0.960`, ICC `0.881 -> 0.883 -> 0.886`, MAE `0.130 -> 0.125 -> 0.123`
- ST2: r `0.953 -> 0.950 -> 0.953`, ICC `0.881 -> 0.901 -> 0.882`, MAE `0.174 -> 0.148 -> 0.171`

### 5) Delta-only exports
Generated:
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/all_metric_deltas_gapclf_t050.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gapclf_probe_ja/all_metric_deltas_gapclf_t050_nonzero.csv`

## New Script
- `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py`

