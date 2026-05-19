# Event Representation Benchmark

This repository benchmarks event-camera representations across classification,
optical flow, and object detection. It includes both learning-based
representations and traditional event encodings under a shared experimental
structure.

## Methods

Learning-based representations:

- EST
- ERGO
- GET
- Matrix-LSTM
- EvRepSL
- OmniEvent
- Event Pre-training

Traditional representations:

- Event frame / event count
- Binary event image
- Timestamp image
- Time surface
- Voxel grid

## Tasks

- Classification on event-based recognition datasets
- Optical flow on MVSEC
- Object detection on the Prophesee mini detection dataset

## Repository Layout

```text
src/
  datasets/                 Dataset readers and indexed datasets
  detection/                Detection adapters, YOLOv6 utilities, representations
  representations/          Shared representation implementations
  tasks/                    Task-level interfaces
scripts/
  detection/prophesee/      Prophesee detection indexing, caching, training, summary, plots
configs/
  detection/                Detection model configs
docs/                       Run guides and project notes
paper_overleaf/             Paper draft and AAAI template files
optical-flow/               Optical-flow task code and outputs
third_party/                External reference implementations
```

## Object Detection

The maintained detection pipeline now targets the compact Prophesee mini
detection dataset. GEN1-specific detection entrypoints were removed to keep the
server workflow smaller and easier to run.

Minimal smoke test:

```bash
python scripts/detection/prophesee/run_all.py --root /path/to/mini_dataset --methods ergo --epochs 1 --train-limit 8 --val-limit 4 --test-limit 4
```

Full instructions, cached-tensor workflow, and visualization commands are in
`docs/prophesee_mini_detection_guide_zh.md`.

Traditional baseline notes and repo-wide result indexes are in
`docs/traditional_baseline_guide_zh.md` and `docs/traditional_repo_index_zh.md`.

## Notes

Large datasets, checkpoints, caches, and generated outputs should stay outside
Git unless explicitly documented as lightweight artifacts.
