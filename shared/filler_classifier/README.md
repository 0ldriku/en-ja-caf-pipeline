# Filler Classifier (Current + Repro Notes)

This folder contains filler-classifier training and inference utilities used by the EN/JA postprocess step.

## Production Model Used in Current EN/JA Runs

- Model file: `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`
- Inference wrapper: `shared/filler_classifier/filler_model_inference.py`
- Postprocess caller: `en/postprocess_vad_filler_classifier_en.py`

This is the model used in:
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224`
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224`

## Training Code Used

- Script: `shared/filler_classifier/train_podcastfillers_neural_classifier.py`
- Run log: `shared/filler_classifier/model_podcastfillers_neural_v1_full/RUN_LOG.md`
- Model card: `shared/filler_classifier/model_podcastfillers_neural_v1_full/README.md`
- Metrics: `shared/filler_classifier/model_podcastfillers_neural_v1_full/metrics.json`

## Training Setup (Neural)

- Architecture: TC-ResNet8-style temporal CNN
- Input features: log-mel spectrogram (`1 x 64 x 101`), per-clip z-score normalization
- Labels:
  - Positive: `Uh, Um`
  - Negative: `Words, Breath, Laughter, Music`
- Dataset:
  - Metadata CSV: `shared/filler_classifier/podcastfillers_data/PodcastFillers.csv`
  - Episode audio: HuggingFace `ylacombe/podcast_fillers_by_license`
  - Splits used in training command: `CC_BY_3.0,CC_BY_SA_3.0,CC_BY_ND_3.0`

## Command Used for the Current Production Model

```powershell
python shared/filler_classifier/train_podcastfillers_neural_classifier.py `
  --metadata-csv shared/filler_classifier/podcastfillers_data/PodcastFillers.csv `
  --out-dir shared/filler_classifier/model_podcastfillers_neural_v1_full `
  --hf-dataset ylacombe/podcast_fillers_by_license `
  --license-splits CC_BY_3.0,CC_BY_SA_3.0,CC_BY_ND_3.0
```

## References

- Corcoran, K., Nookala, M., Tan, L., et al. (2022). *Filler Word Detection and Classification: A Dataset and Benchmark*. arXiv:2203.15135.
- Dataset project page: https://podcastfillers.github.io/
- Dataset release (metadata/annotation release): Zenodo record associated with the PodcastFillers benchmark.

## Notes on Scope

- This repository currently uses the trained neural checkpoint above for postprocess candidate scoring.
- Candidate extraction and CAF correlation results are reported in EN/JA analysis run logs and reports, not in this README.
