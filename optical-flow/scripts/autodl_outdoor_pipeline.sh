#!/usr/bin/env bash
# Compatibility entrypoint for the current formal MVSEC optical-flow run.
#
# The maintained protocol is implemented in run_mvsec_100e_all_early_stop.sh:
#   train: outdoor_day1 + outdoor_day2
#   eval:  indoor_flying1 + indoor_flying2 + indoor_flying3
#   max epochs: 100
#   early-stop patience: 10
#   validation: block-random
#
# Usage:
#   bash scripts/autodl_outdoor_pipeline.sh
#   bash scripts/autodl_outdoor_pipeline.sh est

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_mvsec_100e_all_early_stop.sh" "$@"
