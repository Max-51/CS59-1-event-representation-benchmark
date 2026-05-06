# MVSEC Traditional vs Learning-based Result Package, 2026-05-06

This folder records a lightweight comparison artifact for the MVSEC optical-flow task.
It contains only derived tables and figures. It does not contain raw MVSEC HDF5/NPZ data
or model checkpoints.

## Source Archives

- traditional: `mvsec_traditional_e100_results_20260506_0301.tar.gz`
- learning-based: `mvsec_original_protocol_e100_earlystop_results_20260501_0141.tar.gz`

## Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- events: 6M extracted left-camera events per sequence
- flow GT: generated full `*_gt_flow_full.npz`
- window size / stride: 200 / 200
- decoder: shared `EVFlowNetLike`
- max epochs: 100
- early stop: patience 10, min delta 0.001
- early-stop validation: 1000 block-random outdoor windows
- metrics: AEE and outlier percent, lower is better

## Result Summary

| Group | Method | AEE | Outlier % | Best epoch | Channels |
|---|---|---:|---:|---:|---:|
| Traditional | Timestamp Image | 2.791623 | 36.608703 | 1 | 2 |
| Learning-based | EST | 2.865447 | 37.039154 | 1 | 18 |
| Traditional | Voxel Grid | 2.900117 | 37.647839 | 13 | 10 |
| Traditional | Binary Event Image | 2.957727 | 38.124872 | 10 | 2 |
| Learning-based | GET | 2.961931 | 38.342230 | 12 | 24 |
| Learning-based | Event Pre-training | 2.965345 | 38.193191 | 10 | 2 |
| Traditional | Event Frame | 2.965373 | 38.185070 | 10 | 2 |
| Learning-based | ERGO | 2.971322 | 38.305344 | 10 | 12 |
| Traditional | Time Surface | 2.976563 | 38.343491 | 10 | 2 |
| Learning-based | MatrixLSTM | 3.013823 | 38.970558 | 12 | 4 |
| Learning-based | EvRepSL | 3.018017 | 39.062013 | 5 | 3 |

## Quick Interpretation

- Best overall AEE in this unified run: `Timestamp Image` (Traditional), AEE 2.791623.
- Best traditional method: `Timestamp Image`, AEE 2.791623.
- Best learning-based method: `EST`, AEE 2.865447.
- The comparison is protocol-aligned: all methods use the same train/eval split,
  shared decoder, early-stop rule, and metrics. It is a unified benchmark comparison,
  not a paper-identical rerun of each method's original private training stack.

## Files

- `tables/mvsec_comparison_results.csv`: one row per method.
- `tables/mvsec_comparison_curves.csv`: epoch-level validation curves.
- `figures/mvsec_aee_bar.png`: AEE comparison.
- `figures/mvsec_outlier_bar.png`: outlier-percent comparison.
- `figures/mvsec_aee_outlier_scatter.png`: AEE/outlier trade-off.
- `figures/mvsec_val_aee_curves.png`: early-stop validation curves.
