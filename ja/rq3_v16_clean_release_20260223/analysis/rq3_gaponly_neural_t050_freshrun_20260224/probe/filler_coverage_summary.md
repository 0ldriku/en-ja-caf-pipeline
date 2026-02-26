# Filler Coverage Summary

- Files analyzed: 40
- Manual TextGrids: `ja/rq3_v16_clean_release_20260223/results/manual_clauses_gold_v2`
- Auto TextGrids: `ja/rq3_v16_clean_release_20260223/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual`
- Classifier candidate intervals: `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/per_file`

## Micro Metrics

| Lexicon | Pred Set | Dur Recall | Dur Precision | Dur F1 | Event Recall | Event Precision |
|---|---:|---:|---:|---:|---:|---:|
| conservative | words_only | 0.194 | 0.588 | 0.292 | 0.312 | 0.560 |
| conservative | words_plus_classifier | 0.239 | 0.383 | 0.295 | 0.358 | 0.489 |
| permissive | words_only | 0.340 | 0.845 | 0.485 | 0.485 | 0.832 |
| permissive | words_plus_classifier | 0.379 | 0.709 | 0.494 | 0.518 | 0.809 |

## Notes

- `conservative` excludes highly ambiguous short forms (`あの/その/あ/え/ん/ま`).
- `permissive` includes those forms as filler tokens.
- `words_plus_classifier` adds predicted filler intervals from per-file classifier CSVs (`pred_filler == 1`).
