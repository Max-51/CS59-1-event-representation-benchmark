# CIFAR10-DVS Traditional Classification Results

Run tag: `20260519_cifar10dvs_aligned_cls_v2_traditional`

Protocol highlights:
- Dataset: `cifar10dvs` (Tonic)
- Split: `random80` (`train=8000`, `test=2000`)
- Selection metric: `test_acc`
- Early stopping: `patience=10`
- Epoch cap: `100`
- Batch size: `32`
- LR: `1e-4`
- Weight decay: `1e-4`
- Device: `NVIDIA GeForce RTX 4090`

Summary:

| method | best acc (%) | epochs |
|---|---:|---:|
| event_frame | 44.95 | 18 |
| binary_event_image | 39.85 | 21 |
| timestamp_image | 41.65 | 16 |
| time_surface | 43.30 | 15 |
| voxel_grid | 43.95 | 18 |

Each method folder contains:
- `config.json`
- `history.jsonl`
- `metrics.json`
- `progress.json`
- `representation_stats.json`
- `result.json`
- `train.log`
