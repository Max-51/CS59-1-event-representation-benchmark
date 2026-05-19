# Prophesee Mini Detection Benchmark Guide

This repository now keeps the object-detection benchmark focused on the
Prophesee mini detection dataset. The dataset root should contain:

```text
mini_dataset/
  label_map_dictionary.json
  train/
    *_td.dat
    *_bbox.npy
    *_bbox.labelfreq
  val/
    *_td.dat
    *_bbox.npy
    *_bbox.labelfreq
  test/
    *_td.dat
    *_bbox.npy
    *_bbox.labelfreq
```

The label files are structured NumPy arrays with `ts`, `x`, `y`, `w`, `h`,
`class_id`, `confidence`, and `track_id`. The default sensor size is
`1280 x 720`; class names are read from `label_map_dictionary.json`.

## Code Layout

```text
src/datasets/prophesee_detection.py              # DAT and bbox readers, window index dataset
src/detection/representations.py                 # Detection-oriented event representations
src/detection/yolov6_common.py                   # YOLOv6 tensor adapter and sample builder
src/detection/yolov6_training.py                 # Shared YOLOv6 train/eval utilities
src/detection/prophesee/yolov6.py                # Prophesee-specific YOLOv6 sample builder
src/detection/prophesee/benchmark.py             # Prophesee datasets and cached tensor dataset
scripts/detection/prophesee/build_window_index.py
scripts/detection/prophesee/cache_tensors.py
scripts/detection/prophesee/train.py
scripts/detection/prophesee/run_all.py
scripts/detection/prophesee/summarize.py
scripts/detection/prophesee/visualize_results.py
configs/detection/yolov6n_prophesee.py
```

The previous GEN1-specific detection entrypoints have been removed. The common
YOLOv6 and representation code is dataset-neutral and is reused by the
Prophesee adapter.

## Smoke Test

Run a small end-to-end check before launching a full server job:

```bash
python scripts/detection/prophesee/run_all.py \
  --root "/root/autodl-tmp/mini_dataset" \
  --methods ergo \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --index-max-files 1 \
  --train-limit 8 \
  --val-limit 4 \
  --test-limit 4 \
  --metadata-dir metadata/prophesee_mini_smoke \
  --output-dir outputs/prophesee_mini_smoke \
  --device cuda \
  --resume
```

## Cached Tensor Workflow

Because this dataset is much smaller than GEN1, it is practical to precompute
YOLOv6-ready tensors and reuse them for repeated training. This avoids slicing
the raw `.dat` files and rebuilding representations every epoch.

```bash
python scripts/detection/prophesee/cache_tensors.py \
  --root "/root/autodl-tmp/mini_dataset" \
  --methods ergo est evrepsl get event_pretraining matrix_lstm \
  --metadata-dir metadata/prophesee_mini_windows \
  --cache-dir cache/prophesee_mini_tensors \
  --img-size 320 \
  --window-us 50000 \
  --cache-dtype float16
```

Then train from the cache:

```bash
python scripts/detection/prophesee/run_all.py \
  --root "/root/autodl-tmp/mini_dataset" \
  --methods ergo est evrepsl get event_pretraining matrix_lstm \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 0 \
  --img-size 320 \
  --window-us 50000 \
  --early-stop-patience 8 \
  --early-stop-metric map50_95 \
  --metadata-dir metadata/prophesee_mini_windows \
  --output-dir outputs/prophesee_mini_detection \
  --cache-dir cache/prophesee_mini_tensors \
  --use-cache \
  --device cuda \
  --resume
```

If memory is stable, increase `--batch-size` to `32`. For methods that build
representations on CUDA, keep `--num-workers 0` unless the method has been
tested with multiprocessing.

## Outputs

Each method writes:

```text
outputs/prophesee_mini_detection/<method>/
  train.log
  metrics.json
  progress.json
  history.jsonl
  checkpoints/
```

Create summary metrics and figures:

```bash
python scripts/detection/prophesee/summarize.py \
  --results-root outputs/prophesee_mini_detection \
  --output outputs/prophesee_mini_detection/summary.json

python scripts/detection/prophesee/visualize_results.py \
  --results-root outputs/prophesee_mini_detection \
  --output-dir outputs/prophesee_mini_detection/figures
```
