#!/usr/bin/env bash
set -euo pipefail

LEARNING_METHODS=("ergo" "est" "event_pretraining" "evrepsl" "get" "matrixlstm")
TRADITIONAL_METHODS=("event_frame" "binary_event_image" "timestamp_image" "time_surface" "voxel_grid")
METHOD_GROUP="${METHOD_GROUP:-learning}"

case "$METHOD_GROUP" in
  learning)
    METHODS=("${LEARNING_METHODS[@]}")
    ;;
  traditional)
    METHODS=("${TRADITIONAL_METHODS[@]}")
    ;;
  all)
    METHODS=("${LEARNING_METHODS[@]}" "${TRADITIONAL_METHODS[@]}")
    ;;
  *)
    echo "ERROR: METHOD_GROUP must be one of: learning, traditional, all" >&2
    exit 2
    ;;
esac
if [[ $# -gt 0 ]]; then
  METHODS=("$@")
fi

DATA_ROOT="${DATA_ROOT:-}"
OUT_DIR="${OUT_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs}"
CURVE_DIR="${CURVE_DIR:-logs/curves}"
PACKAGE_DIR="${PACKAGE_DIR:-.}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.001}"
VAL_WINDOWS="${VAL_WINDOWS:-20}"
VAL_STRATEGY="${VAL_STRATEGY:-block-random}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
TRAIN_TIMESTAMP_SUBWINDOWS_PER_FLOW="${TRAIN_TIMESTAMP_SUBWINDOWS_PER_FLOW:-1}"

if [[ -z "$DATA_ROOT" ]]; then
  cat >&2 <<'EOF'
ERROR: DATA_ROOT is not set.

Set DATA_ROOT to the processed MVSEC folder before running, for example:

  DATA_ROOT=/path/to/mvsec OMP_NUM_THREADS=8 BATCH_SIZE=8 bash scripts/run_mvsec_100e_all_early_stop.sh
  DATA_ROOT=/path/to/mvsec TRAIN_TIMESTAMP_SUBWINDOWS_PER_FLOW=60 VAL_WINDOWS=1000 bash scripts/run_mvsec_100e_all_early_stop.sh ergo
  DATA_ROOT=/path/to/mvsec METHOD_GROUP=traditional bash scripts/run_mvsec_100e_all_early_stop.sh

Expected files include:
  outdoor_day/outdoor_day1_left_events_6m.h5
  outdoor_day/outdoor_day1_gt_flow_full.npz
  outdoor_day/outdoor_day2_left_events_6m.h5
  outdoor_day/outdoor_day2_gt_flow_full.npz
  indoor_flying1/indoor_flying1_left_events_6m.h5
  indoor_flying/indoor_flying1_gt_flow_2000.npz
  indoor_flying/indoor_flying2_left_events_6m.h5
  indoor_flying/indoor_flying2_gt_flow_full.npz
  indoor_flying/indoor_flying3_left_events_6m.h5
  indoor_flying/indoor_flying3_gt_flow_full.npz
EOF
  exit 2
fi

# Prefer the corrected indoor_flying1 generated GT file unless explicitly overridden.
IF1_FLOW="${IF1_FLOW:-$DATA_ROOT/indoor_flying/indoor_flying1_gt_flow_2000.npz}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CURVE_DIR"

for method in "${METHODS[@]}"; do
  echo "===== running ${method}: max ${EPOCHS} epochs, early-stop patience ${PATIENCE}, val ${VAL_STRATEGY} ====="
  wandb_args=()
  if [[ -n "$WANDB_PROJECT" ]]; then
    wandb_args=(
      --wandb-project "$WANDB_PROJECT"
      --wandb-run-name "mvsec_${method}_e${EPOCHS}_${VAL_STRATEGY}"
      --wandb-mode "$WANDB_MODE"
    )
  fi

  python scripts/run_original_protocol.py \
    --adapter "$method" \
    --train-pair "$DATA_ROOT/outdoor_day/outdoor_day1_left_events_6m.h5:$DATA_ROOT/outdoor_day/outdoor_day1_gt_flow_full.npz" \
    --train-pair "$DATA_ROOT/outdoor_day/outdoor_day2_left_events_6m.h5:$DATA_ROOT/outdoor_day/outdoor_day2_gt_flow_full.npz" \
    --eval-pair "$DATA_ROOT/indoor_flying1/indoor_flying1_left_events_6m.h5:$IF1_FLOW" \
    --eval-pair "$DATA_ROOT/indoor_flying/indoor_flying2_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying2_gt_flow_full.npz" \
    --eval-pair "$DATA_ROOT/indoor_flying/indoor_flying3_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying3_gt_flow_full.npz" \
    --epochs "$EPOCHS" \
    --window-alignment timestamp \
    --train-timestamp-subwindows-per-flow "$TRAIN_TIMESTAMP_SUBWINDOWS_PER_FLOW" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --disable-cudnn \
    --progress-every "$PROGRESS_EVERY" \
    --early-stop-patience "$PATIENCE" \
    --early-stop-min-delta "$MIN_DELTA" \
    --early-stop-val-windows "$VAL_WINDOWS" \
    --early-stop-val-strategy "$VAL_STRATEGY" \
    --curve-log "$CURVE_DIR/original_od12_if123_full6m_${method}_e${EPOCHS}_${VAL_STRATEGY}_curve.csv" \
    "${wandb_args[@]}" \
    --output "$OUT_DIR/original_od12_if123_full6m_${method}_e${EPOCHS}_${VAL_STRATEGY}_earlystop.json" \
    2>&1 | tee "$LOG_DIR/original_od12_if123_full6m_${method}_e${EPOCHS}_${VAL_STRATEGY}_earlystop.log"

  echo "===== done ${method} ====="
done

PACKAGE_PATH="$PACKAGE_DIR/mvsec_eventvalid_timestamp_${METHOD_GROUP}_tw${TRAIN_TIMESTAMP_SUBWINDOWS_PER_FLOW}_vw${VAL_WINDOWS}_bs${BATCH_SIZE}_e${EPOCHS}_earlystop_results_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "$PACKAGE_PATH" "$OUT_DIR" "$LOG_DIR" docs README.md README_FOR_GROUP.md
echo "===== all done ====="
ls -lh "$PACKAGE_PATH"
