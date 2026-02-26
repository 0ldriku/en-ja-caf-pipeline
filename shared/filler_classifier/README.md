# Filler Classifier (Current release)

This folder contains training and inference utilities for the shared EN/JA filler-candidate scoring step.

## Production model used

- Model: `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`
- Inference wrapper: `shared/filler_classifier/filler_model_inference.py`
- Postprocess caller: `en/postprocess_vad_filler_classifier_en.py`

Used in final fresh-run folders:
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/`

## Training code

- `shared/filler_classifier/train_podcastfillers_neural_classifier.py`
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/RUN_LOG.md`
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/metrics.json`

## Reference

- Corcoran et al. (2022), arXiv:2203.15135
- https://podcastfillers.github.io/
