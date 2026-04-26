# Optical Flow Folder Guide

This folder is the optical-flow part of the benchmark group work.

## What Is Inside

- `src/`
  The benchmark package code, including MVSEC loading, adapters, metrics, and
  the shared optical-flow training/evaluation pipeline.
- `scripts/`
  End-to-end scripts for smoke tests, MVSEC event conversion, GT flow
  generation, single-method runs, and the formal outdoor-train / indoor-test
  AutoDL helper.
- `docs/`
  Handoff notes, environment notes, adapter status, and the current review of
  scientific limitations.
- `configs/envs/`
  Per-method dependency lists.
- `artifacts/mvsec_results/20260426/`
  The lightweight formal result package that can be kept in GitHub.

## Current Result Scope

The current main result is:

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- flow GT: generated full `*_gt_flow_full.npz`
- decoder: shared `EVFlowNetLike`
- epochs: `1`

This is a unified adapted reproduction benchmark, not a paper-identical rerun
of every original optical-flow codebase.

## Result Archive

Tracked artifact:

```text
artifacts/mvsec_results/20260426/mvsec_original_protocol_results_20260426.tar.gz
```

That archive contains only result JSONs, logs, and documentation snapshots. It
does not contain raw MVSEC data.

## Most Important Entry Points

- `README.md`
- `docs/MVSEC_TASK_REVIEW_20260426.md`
- `docs/CLAUDE_CODE_HANDOFF.md`
- `scripts/autodl_outdoor_pipeline.sh`
- `scripts/run_original_protocol.py`
