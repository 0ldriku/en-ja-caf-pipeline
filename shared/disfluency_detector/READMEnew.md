# Shared Disfluency Detector

Canonical shared location for the EN/JA disfluency model used by clause segmentation scripts.

## Current model

- `model_v2/final/` (copied from `en/disfluency_test/l2_disfluency_detector/model_v2/final`)

## Notes

- Only `v2` is mirrored here as the production model.
- EN/JA segmenter scripts now resolve this shared path first, then fall back to the legacy EN path.

## Training provenance (model_v2)

- This shared copy is a runtime mirror only.
- Full training methodology, dataset prep, and evaluation are documented in:
  - `en/disfluency_test/l2_disfluency_detector/README.md`
  - `en/disfluency_test/l2_disfluency_detector/PROGRESS.md`
- Training/processing scripts are in:
  - `en/disfluency_test/l2_disfluency_detector/scripts/`
    - `download_source_data.py`
    - `inject_english.py`
    - `inject_japanese.py`
    - `prepare_labels.py`
    - `download_switchboard.py`
    - `train.py`
