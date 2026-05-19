# GPU Rerun Instructions

This is the handoff checklist for rerunning the MVSEC optical-flow benchmark on
any Linux machine with a CUDA GPU. The data folder is not stored in GitHub, so
the runner needs `DATA_ROOT` to point to the processed MVSEC folder on that
machine.

## 1. Update Code

```bash
git clone https://github.com/Max-51/CS59-1-event-representation-benchmark.git
cd CS59-1-event-representation-benchmark/optical-flow
```

If the repository is already cloned:

```bash
cd CS59-1-event-representation-benchmark/optical-flow
git pull --rebase origin main
```

The updated runner uses:

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- default learning methods: `ergo`, `est`, `event_pretraining`, `evrepsl`,
  `get`, `matrixlstm`
- optional traditional methods: `event_frame`, `binary_event_image`,
  `timestamp_image`, `time_surface`, `voxel_grid`
- shared decoder: `EVFlowNetLike`
- max epochs: 100
- batch size: 8
- early-stop patience: 10
- default validation: 20 block-random outdoor windows for the small timestamp
  sanity run
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps
- optional training densification: split each outdoor timestamp interval into
  multiple training subwindows while keeping evaluation unchanged
- metrics: event-valid AEE / KITTI-style Outlier, evaluated only on pixels that
  fired at least one event in the corresponding event window

## 2. Prepare Data

Use the processed MVSEC folder, not the raw dataset folder. The runner expects
this layout under `DATA_ROOT`:

```text
DATA_ROOT/
├── outdoor_day/
│   ├── outdoor_day1_left_events_6m.h5
│   ├── outdoor_day1_gt_flow_full.npz
│   ├── outdoor_day2_left_events_6m.h5
│   └── outdoor_day2_gt_flow_full.npz
├── indoor_flying1/
│   └── indoor_flying1_left_events_6m.h5
└── indoor_flying/
    ├── indoor_flying1_gt_flow_2000.npz
    ├── indoor_flying2_left_events_6m.h5
    ├── indoor_flying2_gt_flow_full.npz
    ├── indoor_flying3_left_events_6m.h5
    └── indoor_flying3_gt_flow_full.npz
```

Important: the event HDF5 files must be generated with the current converter.
Older processed event files may store Unix-time event timestamps as float32,
which collapses sub-second timing. If the checker reports float32 timestamps or
many identical adjacent timestamps, regenerate the event `.h5` files from the
raw `.bag` files before running the benchmark.

The important corrected file is `indoor_flying1_gt_flow_2000.npz`. It should
have 1398 flow frames. Check it with:

```bash
DATA_ROOT=/path/to/mvsec
python - <<'PY'
import os
import numpy as np

f = os.path.join(os.environ["DATA_ROOT"], "indoor_flying", "indoor_flying1_gt_flow_2000.npz")
with np.load(f) as d:
    print(d["x_flow_dist"].shape)
    print(len(d["timestamps"]))
PY
```

Expected output:

```text
(1398, 260, 346)
1398
```

## 3. Optional Alignment Check

```bash
DATA_ROOT=/path/to/mvsec
python scripts/check_mvsec_alignment.py --data-root "$DATA_ROOT"
```

This prints the event timestamp range and the flow timestamp coverage for each
sequence. It is a quick sanity check before launching the long run.

## 4. Run the Learning Benchmark

Small sanity run, matching the one-window-per-flow timestamp setting:

```bash
DATA_ROOT=/path/to/mvsec \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh
```

Training-densified probe, intended to test whether the small 149-window training
set is the main bottleneck:

```bash
DATA_ROOT=/path/to/mvsec \
TRAIN_TIMESTAMP_SUBWINDOWS_PER_FLOW=60 \
VAL_WINDOWS=1000 \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh ergo
```

This keeps evaluation as one timestamp-aligned window per indoor flow frame. Only
the outdoor training side is densified. With the current MVSEC files this should
create roughly 10k outdoor training/validation candidates before early-stop
splitting.

The runner writes:

- per-method JSON results under `results/`
- per-method logs under `logs/`
- per-method curve CSV files under `logs/curves/`
- a packaged result archive in the current `optical-flow/` folder

The JSON field `metric_scope` should be `event_valid`.

If GPU memory is tight, keep `BATCH_SIZE=8` first and only lower it if the run
fails with CUDA out-of-memory.

To run the traditional methods under the same timestamp-aligned protocol:

```bash
DATA_ROOT=/path/to/mvsec \
METHOD_GROUP=traditional \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh
```

## 5. Build Tables and Figures

After the benchmark finishes:

```bash
python scripts/build_mvsec_e100_outputs.py \
  --results-dir results \
  --curve-dir logs/curves \
  --summary-dir results/summary \
  --figures-dir results/figures
```

For traditional-only results, add:

```bash
--method-group traditional
```

Expected generated deliverables:

- `results/summary/mvsec_e100_earlystop_summary.csv`
- `results/summary/mvsec_e100_earlystop_summary.md`
- `results/figures/mvsec_e100_earlystop_aee.svg`
- `results/figures/mvsec_e100_earlystop_outlier.svg`
- `results/figures/mvsec_e100_earlystop_train_loss_curve.svg`
- `results/figures/mvsec_e100_earlystop_val_curve.svg`

Send back the generated `.tar.gz` package, or send back `results/`, `logs/`,
and `logs/curves/` after checking the run completed.
