# PodcastFillers Neural Classifier Run Log

## Command
```powershell
python shared/filler_classifier/train_podcastfillers_neural_classifier.py \
  --metadata-csv shared\filler_classifier\podcastfillers_data\PodcastFillers.csv \
  --out-dir shared\filler_classifier\model_podcastfillers_neural_v1_full \
  --hf-dataset ylacombe/podcast_fillers_by_license \
  --license-splits CC_BY_3.0,CC_BY_SA_3.0,CC_BY_ND_3.0
```

## Label Setup
- Positive: Uh,Um
- Negative: Words,Breath,Laughter,Music

## Feature Setup
- target_sr: 16000
- n_mels: 64
- win_ms: 25.0
- hop_ms: 10.0
- max_frames: 101

## Model/Train Setup
- Architecture: TC-ResNet8-style 1D temporal Conv-ResNet
- batch_size: 64
- epochs: 30
- lr: 0.001
- early_stop_patience: 5
- lr_patience: 3
- lr_factor: 0.5
- best_epoch: 10

## Extraction Stats
- metadata_events_requested: 76689
- episodes_requested: 199
- episodes_matched: 199
- episode_decode_fail: 0
- event_bounds_fail: 0
- feature_fail: 0
- events_extracted: 76689

## Metrics
- validation F1: 0.9409, P: 0.9686, R: 0.9146, AUC: 0.9798066499290965
- test F1: 0.9328, P: 0.9489, R: 0.9171, AUC: 0.9796474616245466
