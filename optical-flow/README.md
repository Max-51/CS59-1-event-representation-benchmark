# MVSEC Optical Flow Benchmark

This folder contains the optical-flow part of the COMP5703 benchmark group
work. It implements a unified downstream benchmark on MVSEC for runnable event
representations, all evaluated with the same EVFlowNet-like decoder.

This is an adapted reproduction benchmark. It is not a bit-for-bit rerun of
each paper's original optical-flow decoder, training stack, or private
downstream head.

## Current Status

- Dataset: MVSEC optical-flow sequences.
- Train split: `outdoor_day1 + outdoor_day2`.
- Test split: `indoor_flying1/2/3`.
- Default learning methods: `ergo`, `est`, `event_pretraining`, `evrepsl`, `get`,
  `matrixlstm`.
- Optional traditional methods under the same protocol: `event_frame`,
  `binary_event_image`, `timestamp_image`, `time_surface`, `voxel_grid`.
- Paper-reference only: OmniEvent, not a local runnable method.
- Decoder: shared `EVFlowNetLike`.
- Training protocol: max 100 epochs, batch size 8, early-stop patience 10.
- Validation: block-random validation sampled from outdoor training windows.
- Metrics: AEE/EPE and KITTI-style outlier percentage.
- Event/flow pairing: timestamp-aligned event intervals from flow GT
  timestamps.
- Data correction: event HDF5 timestamps must be stored with float64 precision.
  Older processed event `.h5` files written with float32 Unix timestamps are not
  suitable for the formal rerun.
- Current result state: the corrected float64/timestamp-aligned rerun completed
  on 2026-05-16. Result JSON, curves, summaries, and figures are committed under
  `results_float64_cached_20260516/` and `logs_float64_cached_20260516/curves/`.

## Latest Result Snapshot

The current formal result uses `mvsec_float64_delivery_20260516`, float64 event
timestamps, timestamp-aligned event/flow windows, batch size 8, 100 epoch cap,
and early-stop patience 10. All 11 local methods record
`window_alignment: "timestamp"`, with `train_windows=149` and
`eval_windows=871`.

| Group | Best method | AEE | Outlier % |
| --- | --- | ---: | ---: |
| Learning-based | EST | 2.0429 | 23.36 |
| Traditional | Voxel Grid | 2.0759 | 23.85 |
| Overall | EST | 2.0429 | 23.36 |

Full tables:

- `results_float64_cached_20260516/summary_all/mvsec_e100_earlystop_summary.md`
- `results_float64_cached_20260516/summary_learning/mvsec_e100_earlystop_summary.md`
- `results_float64_cached_20260516/summary_traditional/mvsec_e100_earlystop_summary.md`

Cross-task comparison artifacts are in
`../artifacts/traditional_baseline_analysis/20260516_float64/`.

## Rerun Protocol

The formal 2026-05-16 run used:

- train: `outdoor_day1 + outdoor_day2`
- evaluate: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- flow GT: generated flow files with timestamps
- `indoor_flying1` GT: corrected 1398-frame
  `indoor_flying1_gt_flow_2000.npz`
- event HDF5: regenerated with the current converter so the timestamp column is
  float64, not float32
- shared decoder: `src/mvsec_benchmark/models/evflownet_like.py`
- method groups: learning (`ergo`, `est`, `event_pretraining`, `evrepsl`,
  `get`, `matrixlstm`) and traditional (`event_frame`, `binary_event_image`,
  `timestamp_image`, `time_surface`, `voxel_grid`)

Before launching the long run, use `scripts/check_mvsec_alignment.py` to check
that each event file covers the corresponding flow timestamps and that event
timestamps are not collapsed. This is the main guard against repeating the
earlier wrong-result problem.

```bash
python scripts/check_mvsec_alignment.py --data-root /path/to/processed/mvsec
```

To rerun the same all-method protocol:

```bash
DATA_ROOT=/path/to/processed/mvsec \
METHOD_GROUP=all \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh
```

To rerun the traditional methods under the same timestamp-aligned protocol:

```bash
DATA_ROOT=/path/to/processed/mvsec \
METHOD_GROUP=traditional \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh
```

After the run finishes, rebuild the table and figures:

```bash
python scripts/build_mvsec_e100_outputs.py \
  --results-dir results \
  --curve-dir logs/curves \
  --summary-dir results/summary \
  --figures-dir results/figures \
  --method-group all
```

For traditional-only or learning-only outputs, use `--method-group traditional`
or `--method-group learning`.

## Reporting Notes

- Describe this as a unified downstream optical-flow benchmark / adapted
  reproduction.
- Runnable local event representations use the same EVFlowNet-like decoder and
  the same train/eval split.
- Do not directly compare the local numbers as official-paper
  reproduction numbers, because several papers do not release the same
  optical-flow downstream code.
- Treat `OmniEvent✳` as reported-only context, not as a local run in this
  pipeline.
- Raw MVSEC data and processed `.h5` / `.npz` data are not committed to GitHub.

## Code Layout

```text
optical-flow/
├── configs/
│   └── envs/
├── docs/
├── scripts/
├── src/
│   └── mvsec_benchmark/
└── tests/
```

Key entry points:

- `scripts/run_original_protocol.py`
- `scripts/run_mvsec_100e_all_early_stop.sh`
- `scripts/check_mvsec_alignment.py`
- `scripts/build_mvsec_e100_outputs.py`
- `src/mvsec_benchmark/pipeline.py`
- `src/mvsec_benchmark/models/evflownet_like.py`
- `docs/PROJECT_STATUS.md`
- `docs/GPU_RERUN_INSTRUCTIONS.md`

## Local Smoke Test

The smoke tests use synthetic/mock data and do not require MVSEC downloads.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_smoke.py
```
