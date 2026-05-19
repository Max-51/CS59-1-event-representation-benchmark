from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib_cache_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "traditional_baseline_analysis" / "20260506"
CLASSIFICATION = ROOT / "artifacts" / "traditional_classification" / "tables" / "traditional_classification_results.csv"
MVSEC = ROOT / "optical-flow" / "artifacts" / "mvsec_results" / "20260506_traditional_vs_learning" / "tables" / "mvsec_comparison_results.csv"

METHOD_LABELS = {
    "event_frame": "Event Frame",
    "binary_event_image": "Binary Event Image",
    "timestamp_image": "Timestamp Image",
    "time_surface": "Time Surface",
    "voxel_grid": "Voxel Grid",
    "ergo": "ERGO",
    "est": "EST",
    "event_pretraining": "Event Pre-training",
    "evrepsl": "EvRepSL",
    "get": "GET",
    "matrixlstm": "MatrixLSTM",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_mvsec_horizontal_bar(rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda row: float(row["aee"]), reverse=True)
    labels = [row["method_label"] for row in rows]
    values = [float(row["aee"]) for row in rows]
    groups = [row["group"] for row in rows]
    colors = ["#E69F00" if group == "Traditional" else "#0072B2" for group in groups]

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ypos = np.arange(len(rows))
    bars = ax.barh(ypos, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AEE (lower is better)")
    ax.set_title("MVSEC Optical Flow: Unified Protocol")
    ax.set_xlim(2.74, 3.05)
    ax.grid(axis="x", color="#D0D0D0", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8.5)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#E69F00", label="Traditional"),
        plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="Learning-based"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "mvsec_aee_horizontal_bar.pdf")
    fig.savefig(OUT / "mvsec_aee_horizontal_bar.png", dpi=240)
    plt.close(fig)


def normalize_high(values: dict[str, float]) -> dict[str, float]:
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def normalize_low(values: dict[str, float]) -> dict[str, float]:
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return {key: 1.0 for key in values}
    return {key: (hi - value) / (hi - lo) for key, value in values.items()}


def save_cross_task_heatmap(class_rows: list[dict[str, str]], mvsec_rows: list[dict[str, str]]) -> None:
    methods = ["event_frame", "binary_event_image", "timestamp_image", "time_surface", "voxel_grid"]
    nmnist = {row["method"]: float(row["test_accuracy"]) for row in class_rows if row["dataset"] == "nmnist"}
    ncaltech = {row["method"]: float(row["test_accuracy"]) for row in class_rows if row["dataset"] == "ncaltech101"}
    mvsec = {row["method"]: float(row["aee"]) for row in mvsec_rows if row["group"] == "Traditional"}

    score_maps = [
        normalize_high(nmnist),
        normalize_high(ncaltech),
        normalize_low(mvsec),
    ]
    raw_maps = [nmnist, ncaltech, mvsec]
    columns = ["N-MNIST Acc.", "N-Caltech101 Acc.", "MVSEC AEE"]
    matrix = np.array([[score_maps[col][method] for col in range(3)] for method in methods])

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[method] for method in methods])
    ax.set_title("Traditional Representations Across Tasks")
    for i, method in enumerate(methods):
        for j in range(3):
            raw = raw_maps[j][method]
            text = f"{raw * 100:.2f}%" if j < 2 else f"{raw:.3f}"
            color = "white" if matrix[i, j] < 0.45 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Within-task normalized score")
    fig.tight_layout()
    fig.savefig(OUT / "traditional_cross_task_heatmap.pdf")
    fig.savefig(OUT / "traditional_cross_task_heatmap.png", dpi=240)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    class_rows = read_rows(CLASSIFICATION)
    mvsec_rows = read_rows(MVSEC)
    save_mvsec_horizontal_bar(mvsec_rows)
    save_cross_task_heatmap(class_rows, mvsec_rows)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
