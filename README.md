# EN+JA Unified Release Bundle (2026-02-24)

This folder packages the current production releases and shared models into one portable unit.

## Included

- `en/rq123_clean_release_20260223/`
- `ja/rq3_v16_clean_release_20260223/`
- `shared/disfluency_detector/model_v2/final/`
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/`
- Minimal shared helper scripts/docs used by both pipelines:
  - `shared/filler_classifier/README.md`
  - `shared/filler_classifier/candidate_review_textgrids.py`
  - `shared/filler_classifier/filler_model_inference.py`
  - `shared/filler_classifier/train_podcastfillers_neural_classifier.py`
  - `shared/disfluency_detector/README.md`

## Why this bundle

- One folder for transfer/archive
- All key EN + JA release assets together
- Shared model dependencies colocated
- Non-final temp/sweep artifacts were removed; remaining files are the report-cited and rerun-relevant assets.

## Git usage

From this folder:

```powershell
git init
git add .
git commit -m "EN+JA release bundle with shared models"
git remote add origin <YOUR_REMOTE_URL>
git push -u origin main
```

If your remote enforces large-file storage, install/configure Git LFS before pushing.
