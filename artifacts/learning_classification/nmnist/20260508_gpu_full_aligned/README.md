# N-MNIST GPU Full Aligned Learning Results

These files record the full N-MNIST learning-based classification runs used by the updated comparison report.

- split: official N-MNIST train/test
- device: NVIDIA GeForce RTX 4090
- epochs: 100 maximum
- early stopping: patience 10
- batch size: 32
- learning rate: 0.0001
- weight decay: 0.0001
- seed: 42
- classifier: ResNet18-style EventClassifier used by the repository training script

| Method | Best test accuracy (%) | Best epoch | Actual epochs | Channels |
| --- | ---: | ---: | ---: | ---: |
| Matrix-LSTM | 99.40 | 15 | 25 | 16 |
| GET | 99.33 | 23 | 33 | 12 |
| EST | 99.28 | 16 | 26 | 18 |
| ERGO | 99.24 | 17 | 27 | 12 |
| OmniEvent | 99.00 | 26 | 36 | 8 |
| EvRepSL | 98.59 | 15 | 25 | 3 |
| Event Pre-training | 98.18 | 16 | 26 | 2 |

Raw JSON files are under `records/`. Checkpoints and raw datasets are intentionally excluded.