#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: bash scripts/run_cifar10dvs_classification_benchmark.sh

Environment overrides:
  DATA_ROOT, RUN_TAG, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, NUM_WORKERS,
  PREFETCH_FACTOR, SEED, MAX_EVENTS, PATIENCE, DEVICE, TRAIN_LIMIT, TEST_LIMIT,
  RESUME, LEARNING_METHODS, TRADITIONAL_METHODS.
USAGE
  exit 0
fi
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/cifar10dvs}"
RUN_TAG="${RUN_TAG:-20260515_cifar10dvs_gpu_full_aligned}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
SEED="${SEED:-42}"
MAX_EVENTS="${MAX_EVENTS:-50000}"
PATIENCE="${PATIENCE:-10}"
DEVICE="${DEVICE:-cuda}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
RESUME="${RESUME:-1}"
LEARNING_METHODS_STR="${LEARNING_METHODS:-est ergo event_pretraining matrix_lstm evrepsl get omnievent}"
TRADITIONAL_METHODS_STR="${TRADITIONAL_METHODS:-event_frame binary_event_image timestamp_image time_surface voxel_grid}"

export DATA_ROOT RUN_TAG ROOT_DIR

read -r -a LEARNING_METHODS <<< "$LEARNING_METHODS_STR"
read -r -a TRADITIONAL_METHODS <<< "$TRADITIONAL_METHODS_STR"

mkdir -p "$DATA_ROOT"

echo "[info] CIFAR10-DVS root: $DATA_ROOT"
echo "[info] Run tag: $RUN_TAG"
echo "[info] Learning methods: ${LEARNING_METHODS[*]}"
echo "[info] Traditional methods: ${TRADITIONAL_METHODS[*]}"
echo "[info] Limits: train=${TRAIN_LIMIT:-none} test=${TEST_LIMIT:-none}"

python - <<'PY'
import os
from pathlib import Path

import torch
import tonic

root = Path(os.environ.get("DATA_ROOT", "data/cifar10dvs"))
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
dataset = tonic.datasets.CIFAR10DVS(save_to=str(root))
print("cifar10dvs samples:", len(dataset))
print("sensor_size:", getattr(dataset, "sensor_size", None))
PY

COMMON_LEARNING=(
  --dataset cifar10dvs
  --data_root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --device "$DEVICE"
  --seed "$SEED"
  --max_events "$MAX_EVENTS"
  --patience "$PATIENCE"
)

if [[ -n "$TRAIN_LIMIT" ]]; then
  COMMON_LEARNING+=(--train_limit "$TRAIN_LIMIT")
fi
if [[ -n "$TEST_LIMIT" ]]; then
  COMMON_LEARNING+=(--test_limit "$TEST_LIMIT")
fi

for method in "${LEARNING_METHODS[@]}"; do
  [[ -z "$method" ]] && continue
  echo "[learning] $method"
  python "$ROOT_DIR/tasks/classification/scripts/train_classification.py" \
    --method "$method" \
    "${COMMON_LEARNING[@]}" \
    --checkpoint_dir "$ROOT_DIR/artifacts/classification/learning/cifar10dvs/${RUN_TAG}/${method}/checkpoints" \
    --results_dir "$ROOT_DIR/artifacts/classification/learning/cifar10dvs/${RUN_TAG}/results"
done

COMMON_TRADITIONAL=()
if [[ -n "$TRAIN_LIMIT" ]]; then
  COMMON_TRADITIONAL+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$TEST_LIMIT" ]]; then
  COMMON_TRADITIONAL+=(--test-limit "$TEST_LIMIT")
fi
if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "yes" ]]; then
  COMMON_TRADITIONAL+=(--resume)
fi

for method in "${TRADITIONAL_METHODS[@]}"; do
  [[ -z "$method" ]] && continue
  echo "[traditional] $method"
  python "$ROOT_DIR/tasks/classification/scripts/train_traditional_classification.py" \
    --dataset cifar10dvs \
    --root "$DATA_ROOT" \
    --method "$method" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --seed "$SEED" \
    --max-events "$MAX_EVENTS" \
    --early-stop-patience "$PATIENCE" \
    --device "$DEVICE" \
    "${COMMON_TRADITIONAL[@]}" \
    --output-dir "$ROOT_DIR/artifacts/classification/traditional/cifar10dvs/${RUN_TAG}/${method}"
done

python - <<'PY'
import glob
import json
import os
from pathlib import Path

run_tag = os.environ.get("RUN_TAG", "20260515_cifar10dvs_gpu_full_aligned")
root = Path(os.environ.get("ROOT_DIR", "."))
learning = sorted(glob.glob(str(root / f"artifacts/classification/learning/cifar10dvs/{run_tag}/results/*/*_cifar10dvs.json")))
traditional = sorted(glob.glob(str(root / f"artifacts/classification/traditional/cifar10dvs/{run_tag}/*/metrics.json")))

print("[summary] learning result files:", len(learning))
for path in learning:
    data = json.load(open(path))
    print(Path(path).name, data["best_test_accuracy_pct"], data["early_stopping"])

print("[summary] traditional result files:", len(traditional))
for path in traditional:
    data = json.load(open(path))
    print(Path(path).parent.name, round(data["test"]["accuracy"] * 100, 2), {
        "best_epoch": data["best_epoch"],
        "best_val_accuracy": data["best_val_accuracy"],
        "stopped_early": data.get("stopped_early", False),
    })
PY
