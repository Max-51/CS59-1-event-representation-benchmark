import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(results_root):
    rows = []
    for metrics_path in sorted(Path(results_root).glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = payload.get("test") or {}
        rows.append(
            {
                "method": payload["method"],
                "map50": float(test.get("map50") or 0.0),
                "map50_95": float(test.get("map50_95") or 0.0),
                "precision": float(test.get("precision") or 0.0),
                "recall": float(test.get("recall") or 0.0),
                "seconds": float(payload.get("total_seconds") or 0.0),
                "epochs": int(payload.get("completed_epochs") or 0),
            }
        )
    return rows


def save_bar(rows, key, ylabel, output_path):
    rows = sorted(rows, key=lambda row: row[key], reverse=True)
    labels = [row["method"] for row in rows]
    values = [row[key] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Event representation")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metric_group(rows, output_path):
    labels = [row["method"] for row in rows]
    metrics = ["map50", "map50_95", "precision", "recall"]
    x = range(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric in enumerate(metrics):
        values = [row[metric] for row in rows]
        positions = [idx + (offset - 1.5) * width for idx in x]
        ax.bar(positions, values, width=width, label=metric)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Event representation")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create detection benchmark result plots.")
    parser.add_argument("--results-root", default="outputs/prophesee_mini")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(results_root)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {results_root}")

    save_bar(rows, "map50_95", "mAP50:95", output_dir / "detection_map50_95.png")
    save_bar(rows, "map50", "mAP50", output_dir / "detection_map50.png")
    save_bar(rows, "seconds", "Total seconds", output_dir / "detection_runtime.png")
    save_metric_group(rows, output_dir / "detection_metrics_grouped.png")
    print(f"[done] wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
