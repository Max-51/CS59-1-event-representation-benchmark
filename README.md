# Event Representation Benchmark (CS59)

## Overview
This project focuses on benchmarking **learning-based event representations** for event camera data.

## Currently Included Methods
- EST  
- ERGO  
- GET  
- Matrix-LSTM  
- EvRepSL  
- OmniEvent  
- Event Pre-training  

## Target Tasks
- Classification  
- Optical Flow  
- Object Detection  

## Included Papers

| Method | Paper | Venue | Year | Code |
|--------|------|-------|------|------|
| EST | [End-to-End Learning of Representations for Asynchronous Event-Based Data](https://arxiv.org/abs/1904.08245) | ICCV | 2019 | https://github.com/uzh-rpg/rpg_event_representation_learning |
| ERGO | [From Chaos Comes Order: Ordering Event Representations for Object Recognition and Detection](https://arxiv.org/abs/2310.02642) | ICCV | 2023 | https://github.com/uzh-rpg/event_representation_study |
| GET | [Group Event Transformer for Event-Based Vision](https://arxiv.org/abs/2304.13455) | ICCV | 2023 | https://github.com/Peterande/GET-Group-Event-Transformer |
| Matrix-LSTM | [A Differentiable Recurrent Surface for Asynchronous Event-Based Data](https://arxiv.org/abs/2001.03455) | ECCV | 2020 | https://github.com/marcocannici/matrixlstm |
| EvRepSL | [Event-stream Representation via Self-supervised Learning for Event-Based Vision](https://arxiv.org/abs/2412.07080) | TIP | 2024 | https://github.com/VincentQQu/EvRepSL |
| OmniEvent | [OmniEvent: Unified Event Representation Learning](https://arxiv.org/abs/2508.01842) | AAAI | 2026 | - |
| Event Pre-training | [Event Camera Data Pre-training](https://arxiv.org/abs/2301.01928) | ICCV | 2023 | https://github.com/Yan98/Event-Camera-Data-Pre-training |

## Project Scope
We focus on learning-based representations due to their flexibility, adaptability, and potential for end-to-end optimization.

## Repository Structure

- **data/**
  - dataset organization

- **docs/**
  - documentation and paper summaries

- **metadata/papers/**
  - structured metadata for included papers

- **src/**
  - `datasets/` : dataset interfaces
  - `representations/` : representation wrappers and registry
  - `tasks/` : task interfaces

- **third_party/**
  - external reference implementations (Git submodules)

- **Root Files**
  - `.gitmodules` : submodule configuration
  - `environment.yml` : environment setup (to be completed)
  - `requirements.txt` : dependency list (placeholder)
  - `run_benchmark.py` : benchmark entry script
  - `test_registry.py` : registry testing script
  - `test_evrepsl_local.py` : local EvRepSL testing script

## GEN1 Detection Workflow

The current training path is built around a unified preprocessing index:

1. Build fixed 50 ms GEN1 window metadata once:

```bash
python scripts/build_gen1_window_index.py --root /path/to/detection_dataset_duration_60s_ratio_1.0
```

2. Train one representation method at a time:

```bash
python train_gen1_detection.py --root /path/to/detection_dataset_duration_60s_ratio_1.0 --method ergo
```

3. Or run the full six-method benchmark in one command:

```bash
python run_all_gen1_methods.py \
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

4. Summarize all finished method runs:

```bash
python summarize_gen1_results.py
```

This keeps preprocessing shared, avoids storing duplicated event windows, and writes per-method checkpoints, logs, progress files, and metrics under `outputs/benchmark/`. The full benchmark runner skips methods that already have test metrics unless `--force` is passed.
