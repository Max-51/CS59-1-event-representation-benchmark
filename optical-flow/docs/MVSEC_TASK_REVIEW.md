# MVSEC Optical Flow Task Review

This document records the current optical-flow benchmark state after the
2026-05-16 float64/timestamp-aligned rerun.

## What Is Complete

The repository contains a runnable unified downstream benchmark path for 11
local methods:

- `est`
- `ergo`
- `event_pretraining`
- `evrepsl`
- `get`
- `matrixlstm`
- `event_frame`
- `binary_event_image`
- `timestamp_image`
- `time_surface`
- `voxel_grid`

`OmniEvent✳` is not reproduced locally. It is included only as a paper-reported
reference when needed for comparison discussion.

## Current Formal Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event cap: 6M events per sequence HDF5
- event timestamp precision: float64 required for Unix-time event timestamps
- flow GT: generated flow files with timestamps
- corrected `indoor_flying1` GT: `indoor_flying1_gt_flow_2000.npz`, 1398 frames
- decoder: shared `EVFlowNetLike`
- max epochs: 100
- batch size: 8
- early stopping: patience 10
- validation: block-random outdoor validation
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps
- metrics: AEE/EPE and KITTI-style outlier percentage over valid GT flow
  pixels, matching the committed 2026-05-16 result table.

## Result State

The current formal result table is committed from the 2026-05-16 rerun:

```text
results_float64_cached_20260516/summary_all/mvsec_e100_earlystop_summary.md
```

All 11 local methods were run with `window_alignment: "timestamp"`,
`train_windows=149`, and `eval_windows=871`.

| Group | Best method | AEE | Outlier % |
| --- | --- | ---: | ---: |
| Learning-based | EST | 2.0429 | 23.36 |
| Traditional | Voxel Grid | 2.0759 | 23.85 |
| Overall | EST | 2.0429 | 23.36 |

The older processed event tar files may still be invalid if their HDF5
timestamp column was written as float32. Those files cannot recover sub-second
timing after the fact; regenerate event HDF5 files from raw bags with the
current converter before accepting any future rerun.

The accepted result includes:

- per-method JSON outputs under `results_float64_cached_20260516/`
- curve CSV files under `logs_float64_cached_20260516/curves/`
- rebuilt summary CSV and Markdown under `results_float64_cached_20260516/summary_*`
- rebuilt SVG figures under `results_float64_cached_20260516/figures_*`
- cross-task comparison artifacts under `../artifacts/traditional_baseline_analysis/20260516_float64/`

## Reporting Boundary

The current project should be described as an adapted reproduction / unified
downstream benchmark. It should not be presented as a paper-identical
reproduction of every original optical-flow training stack.

Known limitations:

- shared EVFlowNet-like decoder rather than each paper's original downstream
  decoder/head
- timestamp-aligned event/flow pairing is closer to the MVSEC optical-flow
  setup, but the shared downstream decoder still makes this an adapted
  reproduction rather than a paper-identical rerun
- OmniEvent reported-only, not locally reproduced

## Reproducible Interface

The benchmark uses preprocessed MVSEC event and flow files:

```text
*_left_events_6m.h5
*_gt_flow_full.npz or corrected generated flow npz
```

The raw data files are not committed to GitHub. The repository keeps the
preprocessing scripts, sanity checks, and exact run scripts:

- `scripts/convert_mvsec_bag_events.py`
- `scripts/generate_mvsec_flow_from_gt_bag.py`
- `scripts/check_mvsec_alignment.py`
- `scripts/run_mvsec_100e_all_early_stop.sh`
- `scripts/build_mvsec_e100_outputs.py`
