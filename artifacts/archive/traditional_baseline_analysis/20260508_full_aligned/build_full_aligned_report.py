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
OUT = ROOT / "artifacts" / "traditional_baseline_analysis" / "20260508_full_aligned"

TRAD_TABLE = ROOT / "artifacts" / "traditional_classification" / "tables" / "traditional_classification_results.csv"
TRAD_CIFAR_DIR = (
    ROOT
    / "artifacts"
    / "traditional_classification"
    / "cifar10dvs"
    / "20260519_cifar10dvs_aligned_cls_v2_traditional"
)
NMINST_LEARNING_DIR = (
    ROOT / "artifacts" / "learning_classification" / "nmnist" / "20260508_gpu_full_aligned" / "results"
)
LEARNING_RESULTS_DIR = ROOT / "results" / "classification"
MVSEC_SUMMARY = ROOT / "optical-flow" / "results_float64_cached_20260516" / "summary_all" / "mvsec_e100_earlystop_summary.csv"

TRADITIONAL_METHODS = {
    "event_frame",
    "event_count",
    "binary_event_image",
    "timestamp_image",
    "time_surface",
    "voxel_grid",
}

METHOD_LABELS = {
    "binary_event_image": "Binary Event Image",
    "event_count": "Event Count",
    "event_frame": "Event Frame",
    "timestamp_image": "Timestamp Image",
    "time_surface": "Time Surface",
    "voxel_grid": "Voxel Grid",
    "ergo": "ERGO",
    "est": "EST",
    "est_e2e": "EST End-to-End",
    "event_pretraining": "Event Pre-training",
    "evrepsl": "EvRepSL",
    "get": "GET",
    "matrix_lstm": "Matrix-LSTM",
    "matrixlstm": "Matrix-LSTM",
    "omnievent": "OmniEvent",
}

GROUP_COLORS = {"Traditional": "#D55E00", "Learning-based": "#0072B2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_dataset(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if "nmnist" in key:
        return "nmnist"
    if "ncaltech101" in key:
        return "ncaltech101"
    if "cifar10dvs" in key:
        return "cifar10dvs"
    return key


def normalize_method(raw_method: str, source_path: Path) -> str:
    base = str(raw_method or "").strip().lower().replace(" ", "_")
    if source_path.name == "est_e2e_ncaltech101.json":
        return "est_e2e"
    if "end-to-end" in base:
        return "est_e2e"
    if base == "matrixlstm":
        return "matrix_lstm"
    if base.startswith("est"):
        return "est"
    return base or source_path.parent.name.lower()


def pick_float(data: dict, keys: list[str], default: float | None = None) -> float | None:
    for key in keys:
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                continue
    return default


def pick_int(data: dict, keys: list[str], default: int | None = None) -> int | None:
    val = pick_float(data, keys, None)
    if val is None:
        return default
    return int(val)


def extract_history(data: dict) -> list[dict]:
    history = data.get("train_history")
    if isinstance(history, list):
        return history
    history = data.get("training_log")
    if isinstance(history, list):
        return history
    return []


def extract_early_stopping(data: dict) -> dict:
    payload = data.get("early_stopping")
    if isinstance(payload, dict):
        return payload
    return {}


def best_epoch_from_history(history: list[dict]) -> int:
    best_epoch = 0
    best_acc = float("-inf")
    for row in history:
        if "test_acc" not in row:
            continue
        try:
            acc = float(row["test_acc"])
        except (TypeError, ValueError):
            continue
        if acc > best_acc:
            best_acc = acc
            best_epoch = int(row.get("epoch", 0))
    return best_epoch


def clean_out_dir() -> None:
    if not OUT.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        return
    for path in OUT.iterdir():
        if path.name == "build_full_aligned_report.py":
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            for sub in sorted(path.rglob("*"), reverse=True):
                if sub.is_file():
                    sub.unlink()
                else:
                    sub.rmdir()
            path.rmdir()


def load_traditional_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(TRAD_TABLE):
        dataset = normalize_dataset(row["dataset"])
        method = normalize_method(row["method"], TRAD_TABLE)
        rows.append(
            {
                "dataset": dataset,
                "task": "classification",
                "group": "Traditional",
                "method": method,
                "method_label": METHOD_LABELS.get(method, row["method_label"]),
                "accuracy_pct": round(100.0 * float(row["test_accuracy"]), 4),
                "best_epoch": int(row["best_epoch"]),
                "epochs_completed": int(row["epochs"]),
                "early_stopped": str(row["stopped_early"]),
                "batch_size": int(row["batch_size"]),
                "lr": "",
                "weight_decay": "",
                "seed": 42,
                "device": row["device"],
                "input_channels": int(row["input_channels"]),
                "test_samples": int(row["test_samples"]),
                "protocol": "official/full test split",
                "source": row["records_archive"],
                "note": "Traditional representation + ResNet18 classifier",
            }
        )

    for result_path in sorted(TRAD_CIFAR_DIR.glob("*/result.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        method = normalize_method(data.get("method", result_path.parent.name), result_path)
        best_epoch = 0
        progress_path = result_path.parent / "progress.json"
        if progress_path.exists():
            try:
                progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
                best_epoch = int(progress_data.get("best_epoch") or 0)
            except (json.JSONDecodeError, ValueError, TypeError):
                best_epoch = 0
        rows.append(
            {
                "dataset": "cifar10dvs",
                "task": "classification",
                "group": "Traditional",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "accuracy_pct": round(float(data["best_accuracy_pct"]), 4),
                "best_epoch": best_epoch,
                "epochs_completed": int(data["early_stopping"]["actual_epochs"]),
                "early_stopped": str(data["early_stopping"]["triggered"]),
                "batch_size": int(data["batch_size"]),
                "lr": float(data["lr"]),
                "weight_decay": float(data["weight_decay"]),
                "seed": int(data["seed"]),
                "device": str(data.get("gpu", "")),
                "input_channels": int(data["in_channels"]),
                "test_samples": int(data["test_size"]),
                "protocol": f"{data.get('split_strategy', 'unknown')} split",
                "source": str(result_path.relative_to(ROOT)),
                "note": "Traditional representation + ResNet18 classifier",
            }
        )
    return rows


def load_learning_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    nmnist_paths = sorted(NMINST_LEARNING_DIR.rglob("*_nmnist.json"))
    for path in nmnist_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        method = normalize_method(path.parent.name, path)
        history = extract_history(data)
        early = extract_early_stopping(data)
        rows.append(
            {
                "dataset": "nmnist",
                "task": "classification",
                "group": "Learning-based",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "accuracy_pct": round(float(data["best_test_accuracy_pct"]), 4),
                "best_epoch": best_epoch_from_history(history),
                "epochs_completed": int(early.get("actual_epochs", data.get("epochs", 100))),
                "early_stopped": str(early.get("triggered", False)),
                "batch_size": int(data["batch_size"]),
                "lr": float(data["lr"]),
                "weight_decay": float(data["weight_decay"]),
                "seed": int(data["seed"]),
                "device": str(data.get("gpu", "")),
                "input_channels": int(data["in_channels"]),
                "test_samples": 10000,
                "protocol": "official/full test split",
                "source": str(path.relative_to(ROOT)),
                "note": "Learning-based representation + ResNet18 classifier",
            }
        )

    for path in sorted(LEARNING_RESULTS_DIR.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = normalize_dataset(str(data.get("dataset", "")))
        if dataset not in {"ncaltech101", "cifar10dvs"}:
            continue

        method = normalize_method(path.parent.name, path)
        history = extract_history(data)
        early = extract_early_stopping(data)
        acc = pick_float(data, ["best_test_accuracy_pct"], None)
        if acc is None:
            acc_raw = pick_float(data, ["best_test_accuracy", "best_test_acc"], None)
            if acc_raw is None:
                continue
            acc = 100.0 * acc_raw if acc_raw <= 1.0 else acc_raw

        epochs_completed = pick_int(
            data,
            [
                "epochs_actual",
                "actual_epochs_trained",
                "epochs_trained",
                "epochs_actual_trained",
                "actual_epochs",
                "epochs",
            ],
            0,
        )
        if not epochs_completed:
            epochs_completed = int(early.get("actual_epochs", 0))
        if not epochs_completed:
            epochs_completed = int(data.get("epochs_planned", 0))

        rows.append(
            {
                "dataset": dataset,
                "task": "classification",
                "group": "Learning-based",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "accuracy_pct": round(float(acc), 4),
                "best_epoch": best_epoch_from_history(history),
                "epochs_completed": int(epochs_completed),
                "early_stopped": str(early.get("triggered", early.get("stopped", ""))),
                "batch_size": int(pick_int(data, ["batch_size", "actual_batch_size"], 0)),
                "lr": pick_float(data, ["learning_rate", "lr"], ""),
                "weight_decay": pick_float(data, ["weight_decay"], ""),
                "seed": int(pick_int(data, ["seed"], 42)),
                "device": str(data.get("gpu", "")),
                "input_channels": int(pick_int(data, ["in_channels", "input_channels"], 0)),
                "test_samples": int(pick_int(data, ["test_size"], 0)),
                "protocol": str(data.get("split", "repository recorded split")),
                "source": str(path.relative_to(ROOT)),
                "note": "Repository learning-based classification result",
            }
        )
    return rows


def load_mvsec_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(MVSEC_SUMMARY):
        method = normalize_method(row["method"], MVSEC_SUMMARY)
        group = "Traditional" if method in TRADITIONAL_METHODS else "Learning-based"
        rows.append(
            {
                "dataset": "mvsec",
                "task": "optical_flow",
                "group": group,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "aee": float(row["aee"]),
                "outlier_percent": float(row["outlier_percent"]),
                "epochs_completed": int(float(row["epochs_completed"])),
                "best_epoch": int(float(row["best_epoch"])),
                "best_val_aee": float(row["best_val_aee"]),
                "window_alignment": row.get("window_alignment", ""),
                "source": row.get("file", ""),
            }
        )
    return rows


def style_axis(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, linestyle="--", linewidth=0.7, color="#D0D0D0", alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_accuracy(rows: list[dict[str, object]], dataset: str, filename: str, title: str, xmin_pad: float = 2.0) -> None:
    subset = sorted([r for r in rows if r["dataset"] == dataset], key=lambda r: float(r["accuracy_pct"]))
    if not subset:
        return
    labels = [str(r["method_label"]) for r in subset]
    values = [float(r["accuracy_pct"]) for r in subset]
    colors = [GROUP_COLORS[r["group"]] for r in subset]

    fig_h = max(4.8, 0.36 * len(subset) + 1.2)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    y = np.arange(len(subset))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title(title)
    style_axis(ax)
    ax.set_xlim(max(0.0, min(values) - xmin_pad), min(100.5, max(values) + 1.1))

    for bar, value in zip(bars, values):
        ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8.0)

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["Traditional"], label="Traditional"),
            plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["Learning-based"], label="Learning-based"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.pdf")
    fig.savefig(OUT / f"{filename}.png", dpi=280)
    plt.close(fig)


def plot_nmnist_efficiency(rows: list[dict[str, object]]) -> None:
    subset = [r for r in rows if r["dataset"] == "nmnist" and int(r["best_epoch"]) > 0]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for group, marker in [("Traditional", "o"), ("Learning-based", "s")]:
        group_rows = [r for r in subset if r["group"] == group]
        x = [float(r["best_epoch"]) for r in group_rows]
        y = [float(r["accuracy_pct"]) for r in group_rows]
        sizes = [42 + 8 * max(1, float(r["input_channels"])) for r in group_rows]
        ax.scatter(
            x,
            y,
            s=sizes,
            c=GROUP_COLORS[group],
            marker=marker,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
            label=group,
        )
        for r, xi, yi in zip(group_rows, x, y):
            ax.text(xi + 0.22, yi + 0.015, str(r["method_label"]), fontsize=7.1)
    ax.set_xlabel("Best epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("N-MNIST Accuracy vs. Best Epoch")
    ax.set_ylim(94.0, 100.0)
    style_axis(ax, axis="both")
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "nmnist_accuracy_epoch_scatter.pdf")
    fig.savefig(OUT / "nmnist_accuracy_epoch_scatter.png", dpi=280)
    plt.close(fig)


def plot_mvsec(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["aee"]), reverse=True)
    labels = [str(r["method_label"]) for r in rows]
    values = [float(r["aee"]) for r in rows]
    colors = [GROUP_COLORS[r["group"]] for r in rows]
    xmin = max(0.0, min(values) - 0.05)
    xmax = max(values) + 0.05

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AEE (lower is better)")
    ax.set_title("MVSEC Optical Flow (Unified Protocol)")
    ax.set_xlim(xmin, xmax)
    style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8.0)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["Traditional"], label="Traditional"),
            plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["Learning-based"], label="Learning-based"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUT / "mvsec_aee_full_aligned.pdf")
    fig.savefig(OUT / "mvsec_aee_full_aligned.png", dpi=280)
    plt.close(fig)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def best_row(rows: list[dict[str, object]], dataset: str, group: str | None = None) -> dict[str, object]:
    subset = [r for r in rows if r["dataset"] == dataset]
    if group:
        subset = [r for r in subset if r["group"] == group]
    return max(subset, key=lambda r: float(r["accuracy_pct"]))


def write_report_tex(cls_rows: list[dict[str, object]], mvsec_rows: list[dict[str, object]]) -> None:
    nmnist_best_learning = best_row(cls_rows, "nmnist", "Learning-based")
    nmnist_best_trad = best_row(cls_rows, "nmnist", "Traditional")
    ncal_best_learning = best_row(cls_rows, "ncaltech101", "Learning-based")
    ncal_best_trad = best_row(cls_rows, "ncaltech101", "Traditional")
    cifar_best_learning = best_row(cls_rows, "cifar10dvs", "Learning-based")
    cifar_best_trad = best_row(cls_rows, "cifar10dvs", "Traditional")

    mvsec_best = min(mvsec_rows, key=lambda r: float(r["aee"]))
    mvsec_best_learning = min([r for r in mvsec_rows if r["group"] == "Learning-based"], key=lambda r: float(r["aee"]))
    mvsec_best_trad = min([r for r in mvsec_rows if r["group"] == "Traditional"], key=lambda r: float(r["aee"]))

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=0.82in]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{xeCJK}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{caption}}

\setmainfont{{Times New Roman}}
\setsansfont{{Arial}}
\setCJKmainfont{{Songti SC}}
\setCJKsansfont{{STHeiti}}
\setlength{{\parskip}}{{0.62em}}
\setlength{{\parindent}}{{0pt}}

\title{{Traditional 与 Learning-based 事件表示对比报告（最新结果）}}
\author{{}}
\date{{2026-05-20}}

\begin{{document}}
\sloppy
\maketitle

\paragraph{{Scope.}} 本报告统一汇总仓库内截至 2026-05-20 的最新可用结果：N-MNIST、N-Caltech101、CIFAR10-DVS 分类结果，以及 MVSEC 统一 optical-flow protocol 结果。图表和表格均由同一脚本重新生成，旧图与旧表已清理。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.90\linewidth]{{nmnist_accuracy_full_aligned.pdf}}
  \caption{{N-MNIST 分类准确率对比（最新版）。}}
\end{{figure}}

\paragraph{{N-MNIST.}} 最佳 learning-based 方法为 {latex_escape(str(nmnist_best_learning['method_label']))} ({float(nmnist_best_learning['accuracy_pct']):.2f}\%)，最佳 traditional 方法为 {latex_escape(str(nmnist_best_trad['method_label']))} ({float(nmnist_best_trad['accuracy_pct']):.2f}\%)，差距为 {float(nmnist_best_learning['accuracy_pct']) - float(nmnist_best_trad['accuracy_pct']):.2f} 个百分点。该任务整体接近饱和区间。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.88\linewidth]{{nmnist_accuracy_epoch_scatter.pdf}}
  \caption{{N-MNIST 准确率与最佳 epoch 关系。}}
\end{{figure}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.92\linewidth]{{ncaltech101_accuracy_full_aligned.pdf}}
  \caption{{N-Caltech101 分类准确率对比（最新版）。}}
\end{{figure}}

\paragraph{{N-Caltech101.}} 最佳 learning-based 方法为 {latex_escape(str(ncal_best_learning['method_label']))} ({float(ncal_best_learning['accuracy_pct']):.2f}\%)，最佳 traditional 方法为 {latex_escape(str(ncal_best_trad['method_label']))} ({float(ncal_best_trad['accuracy_pct']):.2f}\%)，差距为 {float(ncal_best_learning['accuracy_pct']) - float(ncal_best_trad['accuracy_pct']):.2f} 个百分点。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.92\linewidth]{{cifar10dvs_accuracy_full_aligned.pdf}}
  \caption{{CIFAR10-DVS 分类准确率对比（最新版）。}}
\end{{figure}}

\paragraph{{CIFAR10-DVS.}} 最佳 learning-based 方法为 {latex_escape(str(cifar_best_learning['method_label']))} ({float(cifar_best_learning['accuracy_pct']):.2f}\%)，最佳 traditional 方法为 {latex_escape(str(cifar_best_trad['method_label']))} ({float(cifar_best_trad['accuracy_pct']):.2f}\%)，差距为 {float(cifar_best_learning['accuracy_pct']) - float(cifar_best_trad['accuracy_pct']):.2f} 个百分点。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.88\linewidth]{{mvsec_aee_full_aligned.pdf}}
  \caption{{MVSEC AEE 对比（统一 protocol，越低越好）。}}
\end{{figure}}

\paragraph{{MVSEC Optical Flow.}} 当前总体最优为 {latex_escape(str(mvsec_best['method_label']))}，AEE={float(mvsec_best['aee']):.4f}。最佳 learning-based 为 {latex_escape(str(mvsec_best_learning['method_label']))} (AEE={float(mvsec_best_learning['aee']):.4f})，最佳 traditional 为 {latex_escape(str(mvsec_best_trad['method_label']))} (AEE={float(mvsec_best_trad['aee']):.4f})。在该统一设置下，learning 与 traditional 的最优结果差距较小。

\paragraph{{Summary.}} 最新结果显示：N-MNIST 上两组方法差距很小；N-Caltech101 与 CIFAR10-DVS 上 learning-based 优势更明显；MVSEC 上 traditional 仍保持竞争力。

\end{{document}}
"""
    (OUT / "traditional_vs_learning_full_aligned_report_cn.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    clean_out_dir()
    OUT.mkdir(parents=True, exist_ok=True)

    trad_rows = load_traditional_rows()
    learning_rows = load_learning_rows()
    cls_rows = trad_rows + learning_rows
    cls_rows = sorted(cls_rows, key=lambda r: (str(r["dataset"]), str(r["group"]), -float(r["accuracy_pct"]), str(r["method"])))

    nmnist_learning = [r for r in learning_rows if r["dataset"] == "nmnist"]
    mvsec_rows = load_mvsec_rows()

    write_csv(OUT / "classification_full_aligned_summary.csv", cls_rows)
    write_csv(OUT / "nmnist_learning_full_aligned_summary.csv", nmnist_learning)
    write_csv(OUT / "mvsec_comparison_summary.csv", mvsec_rows)

    plot_accuracy(cls_rows, "nmnist", "nmnist_accuracy_full_aligned", "N-MNIST Classification (Latest)", xmin_pad=1.2)
    plot_nmnist_efficiency(cls_rows)
    plot_accuracy(cls_rows, "ncaltech101", "ncaltech101_accuracy_full_aligned", "N-Caltech101 Classification (Latest)", xmin_pad=3.0)
    plot_accuracy(cls_rows, "cifar10dvs", "cifar10dvs_accuracy_full_aligned", "CIFAR10-DVS Classification (Latest)", xmin_pad=3.0)
    plot_mvsec(mvsec_rows)
    write_report_tex(cls_rows, mvsec_rows)

    print(f"Wrote refreshed full-aligned report assets to {OUT}")


if __name__ == "__main__":
    main()
