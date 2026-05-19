import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Summarize per-method GEN1 benchmark results.")
    parser.add_argument("--results-root", default="artifacts/detection/gen1/default_run")
    parser.add_argument("--output", default="artifacts/detection/gen1/default_run/summary.json")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = []
    for metrics_path in sorted(results_root.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": payload["method"],
                "best_epoch": payload.get("best_epoch"),
                "best_metric_name": payload.get("best_metric_name"),
                "best_metric": payload.get("best_metric"),
                "best_val_map50": payload.get("best_map50"),
                "best_val_map50_95": payload.get("best_map50_95"),
                "test_map50": payload.get("test", {}).get("map50"),
                "test_map50_95": payload.get("test", {}).get("map50_95"),
                "test_precision": payload.get("test", {}).get("precision"),
                "test_recall": payload.get("test", {}).get("recall"),
                "completed_epochs": payload.get("completed_epochs"),
                "stopped_early": payload.get("stopped_early"),
                "total_seconds": payload.get("total_seconds"),
            }
        )

    summary = {"rows": rows}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['method']:18s} "
            f"epochs={row['completed_epochs']} "
            f"early_stop={row['stopped_early']} "
            f"test_mAP50={row['test_map50']} "
            f"test_mAP50_95={row['test_map50_95']} "
            f"precision={row['test_precision']} "
            f"recall={row['test_recall']}"
        )


if __name__ == "__main__":
    main()
