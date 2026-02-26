# PodcastFillers Neural Model (v1)

Model: TC-ResNet8-style temporal CNN
Input: log-mel spectrogram (1 x 64 x 101), per-clip z-score normalization

Training code:
- `shared/filler_classifier/train_podcastfillers_neural_classifier.py`
- Run details: `shared/filler_classifier/model_podcastfillers_neural_v1_full/RUN_LOG.md`

Data source:
- Metadata CSV: `shared\filler_classifier\podcastfillers_data\PodcastFillers.csv`
- Audio dataset: `ylacombe/podcast_fillers_by_license`
- License splits: `CC_BY_3.0,CC_BY_SA_3.0,CC_BY_ND_3.0`

Labels:
- Positive: `Uh,Um`
- Negative: `Breath,Laughter,Music,Words`

Artifacts:
- `model.pt`
- `metrics.json`
- `RUN_LOG.md`
- `train_history.csv`
- `train_manifest.csv`
- `validation_predictions.csv`
- `test_predictions.csv`

Reference:
- Corcoran et al. (2022), *Filler Word Detection and Classification: A Dataset and Benchmark*, arXiv:2203.15135.
