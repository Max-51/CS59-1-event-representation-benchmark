# Event Representation Benchmark (Task-Oriented Layout)

This repository is organized by **three benchmark tasks**:

1. Classification
2. Object Detection (GEN1)
3. Optical Flow (MVSEC)

Core shared code remains in `src/`.

## Task Entrypoints

- Classification: `tasks/classification/`
- Detection: `tasks/detection/`
- Optical flow: `tasks/optical_flow/`

## Quick Start

### 1) Classification (learning-based)

```bash
python tasks/classification/scripts/train_classification.py \
  --method est \
  --dataset nmnist \
  --data_root /path/to/nmnist \
  --checkpoint_dir artifacts/classification/learning/nmnist/demo/est/checkpoints
```

### 2) Classification (traditional)

```bash
python tasks/classification/scripts/train_traditional_classification.py \
  --dataset ncaltech101 \
  --root /path/to/ncaltech101 \
  --method voxel_grid
```

### 3) Detection (GEN1)

```bash
python tasks/detection/scripts/run_all_gen1_methods.py \
  --root /path/to/detection_dataset_duration_60s_ratio_1.0 \
  --methods ergo est evrepsl get event_pretraining matrix_lstm \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 4 \
  --early-stop-patience 15 \
  --early-stop-metric map50_95 \
  --img-size 320 \
  --lr 0.01 \
  --device cuda \
  --resume
```

### 4) Optical Flow (MVSEC)

```bash
python tasks/optical_flow/scripts/check_mvsec_alignment.py --data-root /path/to/processed/mvsec
```

## Dataset Naming

Primary names:

- `nmnist`
- `ncaltech101`
- `cifar10dvs`

Classification aliases are preserved for compatibility, including:

- `minist`/`n-mnist` -> `nmnist`
- `ncar101`/`n-caltech101` -> `ncaltech101`
- `cifa`/`cifar`/`cifar10-dvs` -> `cifar10dvs`

## Artifacts Layout

- Classification: `artifacts/classification/{learning,traditional}/...`
- Detection: `artifacts/detection/gen1/...`
- Optical flow: `artifacts/optical_flow/mvsec/...`
- Cross-task reports: `artifacts/cross_task_reports/latest/...`
- Historical snapshots: `artifacts/archive/...`

## Compatibility Layer

Legacy entry scripts are still available at repository root:

- `train_classification.py`
- `train_est_e2e_classification.py`
- `train_traditional_classification.py`
- `train_gen1_detection.py`
- `run_all_gen1_methods.py`
- `summarize_gen1_results.py`

Legacy paths are preserved through wrappers/symlinks where practical (`optical-flow/`, `metadata/`, `paper_overleaf/`, and selected `artifacts/` paths).

## Documentation

- Global index (Chinese): `docs/task_index_zh.md`
- Traditional method index: `docs/traditional_repo_index_zh.md`
- CIFAR10-DVS classification guide: `docs/cifar10dvs_classification_guide_zh.md`
