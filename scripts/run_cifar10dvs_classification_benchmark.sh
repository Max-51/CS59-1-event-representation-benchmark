#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/tasks/classification/scripts/run_cifar10dvs_classification_benchmark.sh" "$@"
