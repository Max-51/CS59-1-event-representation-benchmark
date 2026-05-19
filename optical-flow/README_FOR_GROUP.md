# Optical Flow Folder Guide

This folder is the optical-flow part of the benchmark group work. It contains
the MVSEC adapted reproduction benchmark code, the shared downstream pipeline,
and the scripts needed to rerun the six local event-representation methods.

## What Is Inside

- `src/`: benchmark package code, including MVSEC loading, representation
  adapters, metrics, and the shared optical-flow training/evaluation pipeline.
- `scripts/`: MVSEC conversion helpers, alignment checks, the main benchmark
  runner, and post-run table/figure generation.
- `docs/`: project status, environment notes, adapter status, and rerun notes.
- `configs/envs/`: per-method dependency lists.
- `tests/`: synthetic tests for the pipeline and data handling.

The corrected float64/timestamp-aligned rerun completed on 2026-05-16. The
current formal result artifacts are committed under
`results_float64_cached_20260516/` and `logs_float64_cached_20260516/curves/`.

## Current Formal Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- flow GT: generated flow files with timestamps
- important data check: `indoor_flying1_gt_flow_2000.npz` should have 1398
  flow frames
- event HDF5 check: timestamp column should be float64. Older processed event
  files with float32 Unix timestamps should be regenerated from raw bags.
- decoder: shared `EVFlowNetLike`
- training: max 100 epochs, batch size 8, early-stop patience 10
- validation: block-random outdoor validation
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps
- optional probe: `TRAIN_TIMESTAMP_SUBWINDOWS_PER_FLOW` can densify only the
  outdoor training side while keeping indoor evaluation unchanged
- metrics: event-valid AEE / KITTI-style Outlier, evaluated only on pixels that
  fired at least one event in the corresponding event window
- default learning methods: `ergo`, `est`, `event_pretraining`, `evrepsl`,
  `get`, `matrixlstm`
- optional traditional methods under the same protocol:
  `event_frame`, `binary_event_image`, `timestamp_image`, `time_surface`,
  `voxel_grid`

This is a unified downstream optical-flow benchmark / adapted reproduction, not
a paper-identical rerun of every original optical-flow codebase.

## Latest Results

The latest MVSEC run used `mvsec_float64_delivery_20260516`, float64 event
timestamps, timestamp-aligned event/flow windows, batch size 8, 100 epoch cap,
and early-stop patience 10.

| Group | Best method | AEE | Outlier % |
| --- | --- | ---: | ---: |
| Learning-based | EST | 2.0429 | 23.36 |
| Traditional | Voxel Grid | 2.0759 | 23.85 |
| Overall | EST | 2.0429 | 23.36 |

Full result table:

```text
results_float64_cached_20260516/summary_all/mvsec_e100_earlystop_summary.md
```

Cross-task comparison report:

```text
../artifacts/traditional_baseline_analysis/20260516_float64/
```

## Rerun Checks

Run the alignment checker before the long benchmark:

```bash
python scripts/check_mvsec_alignment.py --data-root /path/to/processed/mvsec
```

The checker should confirm that flow timestamps are covered by the event
timestamp range and that event timestamps have enough unique values. It is
designed to run before training and does not require a GPU. It should also make
it obvious if `indoor_flying1` has the old 20-frame GT file instead of the
corrected 1398-frame file.

Main run:

```bash
DATA_ROOT=/path/to/processed/mvsec \
METHOD_GROUP=all \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh
```

Training-densified ERGO probe:

```bash
DATA_ROOT=/path/to/processed/mvsec \
TRAIN_TIMESTAMP_SUBWINDOWS_PER_FLOW=60 \
VAL_WINDOWS=1000 \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh ergo
```

Use this only as an adapted training-sample-density check. It is not a claim
that the original papers used this exact subwindow construction.

Traditional-method rerun under the same timestamp-aligned protocol:

```bash
DATA_ROOT=/path/to/processed/mvsec \
METHOD_GROUP=traditional \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh
```

Post-run table and figure generation:

```bash
python scripts/build_mvsec_e100_outputs.py \
  --results-dir results \
  --curve-dir logs/curves \
  --summary-dir results/summary \
  --figures-dir results/figures \
  --method-group all
```

For traditional-only or learning-only outputs, use:

```bash
--method-group traditional
--method-group learning
```

## Most Important Entry Points

- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/GPU_RERUN_INSTRUCTIONS.md`
- `scripts/check_mvsec_alignment.py`
- `scripts/run_original_protocol.py`
- `scripts/run_mvsec_100e_all_early_stop.sh`
- `scripts/build_mvsec_e100_outputs.py`

## Wording To Use

Use:

```text
MVSEC unified downstream optical-flow benchmark / adapted reproduction.
Six runnable event representations are compared under the same EVFlowNet-like
decoder and the same outdoor-train / indoor-test protocol.
Event windows use timestamp-aligned intervals from flow GT timestamps.
Formal results report event-valid sparse-flow AEE / Outlier.
```

Avoid:

```text
Fully reproduced every paper's original optical-flow decoder/head.
Official paper-number comparison.
Treating OmniEvent✳ as a local run.
```
