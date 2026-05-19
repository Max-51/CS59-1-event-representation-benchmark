from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib_cache_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "traditional_baseline_analysis" / "20260507"
TRAD_CLS = ROOT / "artifacts" / "traditional_classification" / "tables" / "traditional_classification_results.csv"
LEARN_CLS_ROOT = ROOT / "results" / "classification"
MVSEC = ROOT / "optical-flow" / "artifacts" / "mvsec_results" / "20260506_traditional_vs_learning" / "tables" / "mvsec_comparison_results.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_classification_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(TRAD_CLS):
        rows.append(
            {
                "dataset": row["dataset"],
                "group": "Traditional",
                "method": row["method_label"],
                "accuracy_pct": 100.0 * float(row["test_accuracy"]),
                "source": row["records_archive"],
                "note": "ResNet18 traditional baseline",
            }
        )

    json_paths = sorted(LEARN_CLS_ROOT.rglob("*.json"))
    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset_raw = str(data.get("dataset", "")).lower()
        if "nmnist" in dataset_raw or "n-mnist" in dataset_raw:
            dataset = "nmnist"
        elif "ncaltech" in dataset_raw or "n-caltech" in dataset_raw:
            dataset = "ncaltech101"
        else:
            continue

        method = str(data.get("method", path.parent.name))
        if path.name == "est_ncaltech101.json":
            method = "EST adaptation"
        elif path.name == "est_e2e_ncaltech101.json":
            method = "EST End-to-End"
        elif path.name == "event_pretraining_ncaltech101.json":
            method = "Event Pre-training"
        elif path.name == "matrix_lstm_ncaltech101.json":
            method = "Matrix-LSTM"
        elif path.name == "evrepsl_nmnist.json":
            method = "EvRepSL"

        if "best_test_accuracy_pct" in data:
            acc = float(data["best_test_accuracy_pct"])
        elif "best_test_accuracy" in data:
            acc = 100.0 * float(data["best_test_accuracy"])
        elif "best_test_acc" in data:
            acc = float(data["best_test_acc"])
        else:
            continue

        rows.append(
            {
                "dataset": dataset,
                "group": "Learning-based",
                "method": method,
                "accuracy_pct": acc,
                "source": str(path.relative_to(ROOT)),
                "note": "Repository learning-based classification result",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_classification(dataset: str, rows: list[dict[str, object]], filename: str, title: str) -> None:
    subset = [row for row in rows if row["dataset"] == dataset]
    subset = sorted(subset, key=lambda row: float(row["accuracy_pct"]))
    labels = [str(row["method"]) for row in subset]
    values = [float(row["accuracy_pct"]) for row in subset]
    colors = ["#E69F00" if row["group"] == "Traditional" else "#0072B2" for row in subset]

    fig_h = max(4.2, 0.38 * len(subset) + 1.5)
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    y = np.arange(len(subset))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#D0D0D0", alpha=0.75)
    ax.set_axisbelow(True)
    xmin = max(0, min(values) - 3.0)
    xmax = min(100.0, max(values) + 2.0)
    ax.set_xlim(xmin, xmax)
    for bar, value in zip(bars, values):
        ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8.5)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#E69F00", label="Traditional"),
        plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="Learning-based"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.pdf")
    fig.savefig(OUT / f"{filename}.png", dpi=240)
    plt.close(fig)


def plot_mvsec(rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda row: float(row["aee"]), reverse=True)
    labels = [row["method_label"] for row in rows]
    values = [float(row["aee"]) for row in rows]
    colors = ["#E69F00" if row["group"] == "Traditional" else "#0072B2" for row in rows]

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AEE (lower is better)")
    ax.set_title("MVSEC Optical Flow: Unified Protocol")
    ax.set_xlim(2.74, 3.05)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#D0D0D0", alpha=0.75)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8.5)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#E69F00", label="Traditional"),
        plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="Learning-based"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "mvsec_aee_brief.pdf")
    fig.savefig(OUT / "mvsec_aee_brief.png", dpi=240)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cls_rows = load_classification_rows()
    cls_rows = sorted(cls_rows, key=lambda row: (str(row["dataset"]), str(row["group"]), -float(row["accuracy_pct"])))
    write_csv(OUT / "classification_comparison_summary.csv", cls_rows)
    mvsec_rows = read_csv(MVSEC)
    write_csv(OUT / "mvsec_comparison_summary.csv", mvsec_rows)
    plot_classification("nmnist", cls_rows, "nmnist_accuracy_brief", "N-MNIST Classification")
    plot_classification("ncaltech101", cls_rows, "ncaltech101_accuracy_brief", "N-Caltech101 Classification")
    plot_mvsec(mvsec_rows)
    print(f"Wrote brief report figures to {OUT}")


if __name__ == "__main__":
    main()
