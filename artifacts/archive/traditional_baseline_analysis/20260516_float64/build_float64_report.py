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
OUT = ROOT / "artifacts" / "traditional_baseline_analysis" / "20260516_float64"
TRAD_CLS = ROOT / "artifacts" / "traditional_classification" / "tables" / "traditional_classification_results.csv"
NCAL_LEARN_ROOT = ROOT / "results" / "classification"
NMNIST_LEARN_ROOT = ROOT / "artifacts" / "learning_classification" / "nmnist" / "20260508_gpu_full_aligned" / "records"
MVSEC_SUMMARY = ROOT / "optical-flow" / "results_float64_cached_20260516" / "summary_all" / "mvsec_e100_earlystop_summary.csv"

METHOD_LABELS = {
    "binary_event_image": "Binary Event Image",
    "event_frame": "Event Frame",
    "timestamp_image": "Timestamp Image",
    "time_surface": "Time Surface",
    "voxel_grid": "Voxel Grid",
    "ergo": "ERGO",
    "est": "EST",
    "event_pretraining": "Event Pre-training",
    "evrepsl": "EvRepSL",
    "get": "GET",
    "matrix_lstm": "Matrix-LSTM",
    "matrixlstm": "Matrix-LSTM",
    "omnievent": "OmniEvent",
}

LEARNING = {"ergo", "est", "event_pretraining", "evrepsl", "get", "matrixlstm"}
TRADITIONAL = {"event_frame", "binary_event_image", "timestamp_image", "time_surface", "voxel_grid"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def load_traditional_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(TRAD_CLS):
        rows.append(
            {
                "dataset": row["dataset"],
                "task": "classification",
                "group": "Traditional",
                "method": row["method"],
                "method_label": row["method_label"],
                "accuracy_pct": 100.0 * float(row["test_accuracy"]),
                "test_loss": float(row["test_loss"]),
                "best_epoch": int(row["best_epoch"]),
                "epochs_completed": int(row["epochs"]),
                "early_stopped": row["stopped_early"],
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
    return rows


def load_nmnist_learning_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(NMNIST_LEARN_ROOT.rglob("*_nmnist.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        history = data.get("train_history", [])
        best_epoch = 0
        if history:
            best_epoch = int(max(history, key=lambda h: float(h.get("test_acc", 0.0))).get("epoch", 0))
        rows.append(
            {
                "dataset": "nmnist",
                "task": "classification",
                "group": "Learning-based",
                "method": data["method"],
                "method_label": METHOD_LABELS.get(data["method"], data["method"]),
                "accuracy_pct": float(data["best_test_accuracy_pct"]),
                "test_loss": "",
                "best_epoch": best_epoch,
                "epochs_completed": int(data.get("early_stopping", {}).get("actual_epochs", data["epochs"])),
                "early_stopped": str(data.get("early_stopping", {}).get("triggered", False)),
                "batch_size": int(data["batch_size"]),
                "lr": float(data["lr"]),
                "weight_decay": float(data["weight_decay"]),
                "seed": int(data["seed"]),
                "device": data["gpu"],
                "input_channels": int(data["in_channels"]),
                "test_samples": 10000,
                "protocol": "official/full test split; GPU full aligned run",
                "source": str(path.relative_to(ROOT)),
                "note": "Learning-based representation + ResNet18 classifier",
            }
        )
    return rows


def load_ncal_learning_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(NCAL_LEARN_ROOT.rglob("*ncaltech101.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = str(data.get("method", path.parent.name))
        label = METHOD_LABELS.get(method, method)
        if path.name == "est_ncaltech101.json":
            label = "EST adaptation"
        elif path.name == "est_e2e_ncaltech101.json":
            label = "EST End-to-End"
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
                "dataset": "ncaltech101",
                "task": "classification",
                "group": "Learning-based",
                "method": method,
                "method_label": label,
                "accuracy_pct": acc,
                "test_loss": "",
                "best_epoch": "",
                "epochs_completed": int(data.get("epochs_trained", data.get("epochs", 100))),
                "early_stopped": "",
                "batch_size": int(data.get("batch_size", 0)),
                "lr": data.get("lr", ""),
                "weight_decay": data.get("weight_decay", ""),
                "seed": data.get("seed", 42),
                "device": data.get("gpu", ""),
                "input_channels": data.get("in_channels", ""),
                "test_samples": "",
                "protocol": "repository recorded N-Caltech101 result",
                "source": str(path.relative_to(ROOT)),
                "note": "Repository learning-based classification result",
            }
        )
    return rows


def load_mvsec_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(MVSEC_SUMMARY):
        method = row["method"]
        group = "Learning-based" if method in LEARNING else "Traditional"
        if method not in LEARNING and method not in TRADITIONAL:
            continue
        rows.append(
            {
                "dataset": "mvsec",
                "task": "optical_flow",
                "group": group,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "aee": float(row["aee"]),
                "outlier_percent": float(row["outlier_percent"]),
                "non_outlier_percent": float(row["non_outlier_percent"]),
                "best_epoch": int(row["best_epoch"]),
                "best_val_aee": float(row["best_val_aee"]),
                "epochs_completed": int(row["epochs_completed"]),
                "early_stopped": row["early_stopped"],
                "train_windows": int(row["train_windows"]),
                "eval_windows": int(row["eval_windows"]),
                "valid_count": int(row["valid_count"]),
                "window_alignment": row["window_alignment"],
                "protocol": "float64 event timestamps; timestamp-aligned windows; outdoor_day1/2 train, indoor_flying1/2/3 eval",
                "source": row["file"],
                "note": "Shared EVFlowNet-like decoder; adapted reproduction benchmark",
            }
        )
    return sorted(rows, key=lambda r: float(r["aee"]))


def style_axis(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, linestyle="--", linewidth=0.7, color="#D0D0D0", alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_accuracy(rows: list[dict[str, object]], dataset: str, filename: str, title: str, xmin_pad: float = 2.0) -> None:
    subset = sorted([r for r in rows if r["dataset"] == dataset], key=lambda r: float(r["accuracy_pct"]))
    labels = [str(r["method_label"]) for r in subset]
    values = [float(r["accuracy_pct"]) for r in subset]
    colors = ["#D55E00" if r["group"] == "Traditional" else "#0072B2" for r in subset]
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
            plt.Rectangle((0, 0), 1, 1, color="#D55E00", label="Traditional"),
            plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="Learning-based"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.pdf")
    fig.savefig(OUT / f"{filename}.png", dpi=280)
    plt.close(fig)


def plot_nmnist_efficiency(rows: list[dict[str, object]]) -> None:
    subset = sorted([r for r in rows if r["dataset"] == "nmnist"], key=lambda r: str(r["group"]))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for group, color, marker in [("Traditional", "#D55E00", "o"), ("Learning-based", "#0072B2", "s")]:
        group_rows = [r for r in subset if r["group"] == group]
        x = [float(r["best_epoch"]) for r in group_rows]
        y = [float(r["accuracy_pct"]) for r in group_rows]
        sizes = [42 + 8 * float(r["input_channels"]) for r in group_rows]
        ax.scatter(x, y, s=sizes, c=color, marker=marker, alpha=0.88, edgecolor="white", linewidth=0.7, label=group)
        for r, xi, yi in zip(group_rows, x, y):
            ax.text(xi + 0.25, yi + 0.015, str(r["method_label"]), fontsize=7.2)
    ax.set_xlabel("Best epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("N-MNIST Accuracy vs. Early-stop Epoch")
    ax.set_ylim(94.7, 99.75)
    style_axis(ax, axis="both")
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "nmnist_accuracy_epoch_scatter.pdf")
    fig.savefig(OUT / "nmnist_accuracy_epoch_scatter.png", dpi=280)
    plt.close(fig)


def plot_mvsec(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda r: float(r["aee"]), reverse=True)
    labels = [str(r["method_label"]) for r in rows]
    values = [float(r["aee"]) for r in rows]
    colors = ["#D55E00" if r["group"] == "Traditional" else "#0072B2" for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AEE (lower is better)")
    ax.set_title("MVSEC Optical Flow: Float64 Timestamp-aligned Results")
    ax.set_xlim(1.95, 2.58)
    style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8.0)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#D55E00", label="Traditional"),
            plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="Learning-based"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUT / "mvsec_aee_float64.pdf")
    fig.savefig(OUT / "mvsec_aee_float64.png", dpi=280)
    plt.close(fig)


def write_report(cls_rows: list[dict[str, object]], mvsec_rows: list[dict[str, object]]) -> None:
    nmnist = [r for r in cls_rows if r["dataset"] == "nmnist"]
    ncal = [r for r in cls_rows if r["dataset"] == "ncaltech101"]
    best_nmnist = max(nmnist, key=lambda r: float(r["accuracy_pct"]))
    best_nmnist_trad = max([r for r in nmnist if r["group"] == "Traditional"], key=lambda r: float(r["accuracy_pct"]))
    best_nmnist_learning = max([r for r in nmnist if r["group"] == "Learning-based"], key=lambda r: float(r["accuracy_pct"]))
    worst_nmnist = min(nmnist, key=lambda r: float(r["accuracy_pct"]))
    best_ncal = max(ncal, key=lambda r: float(r["accuracy_pct"]))
    best_ncal_trad = max([r for r in ncal if r["group"] == "Traditional"], key=lambda r: float(r["accuracy_pct"]))
    mvsec_best = min(mvsec_rows, key=lambda r: float(r["aee"]))
    mvsec_best_learning = min([r for r in mvsec_rows if r["group"] == "Learning-based"], key=lambda r: float(r["aee"]))
    mvsec_best_trad = min([r for r in mvsec_rows if r["group"] == "Traditional"], key=lambda r: float(r["aee"]))
    mvsec_old_note = "older May 8 MVSEC table used pre-float64/full-window artifacts and is kept only as historical context"

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

\title{{Traditional 与 Learning-based 事件表示对比报告\\基于 2026-05-16 MVSEC float64 重跑结果}}
\author{{}}
\date{{2026-05-16}}

\begin{{document}}
\sloppy
\maketitle

\paragraph{{Protocol Alignment.}} 本版报告保留 N-MNIST 与 N-Caltech101 的已对齐分类结果，并将 MVSEC optical flow 替换为 2026-05-16 的 float64 数据重跑结果。MVSEC 使用 outdoor\_day1/2 训练、indoor\_flying1/2/3 测试，event HDF5 timestamp 为 float64，event/flow window 使用 flow GT timestamps 做 timestamp alignment。所有本地 optical-flow 方法共享同一个 EVFlowNet-like decoder、batch size 8、100 epoch 上限、early stopping patience 10。注意：{mvsec_old_note}。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.90\linewidth]{{nmnist_accuracy_full_aligned.pdf}}
  \caption{{N-MNIST 完整官方测试集准确率对比。蓝色为 learning-based 方法，橙色为 traditional baseline。}}
  \label{{fig:nmnist-acc}}
\end{{figure}}

\paragraph{{N-MNIST Is Saturated But Comparable.}} 在完整 N-MNIST setting 下，最佳方法是 {latex_escape(str(best_nmnist['method_label']))}，test accuracy 为 {float(best_nmnist['accuracy_pct']):.2f}\%。最佳 learning-based 方法 {latex_escape(str(best_nmnist_learning['method_label']))} 为 {float(best_nmnist_learning['accuracy_pct']):.2f}\%，最佳 traditional 方法 {latex_escape(str(best_nmnist_trad['method_label']))} 为 {float(best_nmnist_trad['accuracy_pct']):.2f}\%，差距为 {float(best_nmnist_learning['accuracy_pct']) - float(best_nmnist_trad['accuracy_pct']):.2f} 个百分点。除 Binary Event Image 外，大多数方法集中在 98.18--99.40\% 区间，说明 N-MNIST 已接近饱和，适合验证 pipeline 和基本表示能力，但不适合单独作为强结论来源。最弱方法 {latex_escape(str(worst_nmnist['method_label']))} 为 {float(worst_nmnist['accuracy_pct']):.2f}\%。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.88\linewidth]{{nmnist_accuracy_epoch_scatter.pdf}}
  \caption{{N-MNIST 准确率与最佳 epoch 的关系。点大小随输入通道数变化，用于展示表示维度、收敛位置和准确率之间的权衡。}}
  \label{{fig:nmnist-eff}}
\end{{figure}}

\paragraph{{Accuracy-Efficiency Trade-off.}} N-MNIST 的 best epoch 显示出比最终准确率更有区分度的模式。Learning-based 方法大多在第 15--26 epoch 取得最佳结果，其中 Matrix-LSTM 在第 15 epoch 达到 99.40\%，GET 在第 23 epoch 达到 99.33\%，EST 在第 16 epoch 达到 99.28\%。Traditional 方法中 Timestamp Image 的最佳 epoch 仅为 6，准确率仍有 98.05\%；Voxel Grid 在第 12 epoch 达到 99.08\%。这说明更复杂或多通道的表示未必总是需要更长训练，但在 N-MNIST 这种饱和任务中，小幅准确率提升通常伴随更高表示维度或更复杂 adapter。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.92\linewidth]{{ncaltech101_accuracy_full_aligned.pdf}}
  \caption{{N-Caltech101 分类准确率对比。该任务比 N-MNIST 更能拉开 representation capacity 的差异。}}
  \label{{fig:ncaltech}}
\end{{figure}}

\paragraph{{N-Caltech101 Reveals Larger Capacity Gaps.}} N-Caltech101 的结果与 N-MNIST 不同。当前最佳 learning-based 方法 {latex_escape(str(best_ncal['method_label']))} 为 {float(best_ncal['accuracy_pct']):.2f}\%，最佳 traditional 方法 {latex_escape(str(best_ncal_trad['method_label']))} 为 {float(best_ncal_trad['accuracy_pct']):.2f}\%，差距达到 {float(best_ncal['accuracy_pct']) - float(best_ncal_trad['accuracy_pct']):.2f} 个百分点。这个差距比 N-MNIST 明显得多，说明在类别数更多、形状变化更复杂的分类任务上，learning-based representation 更能利用事件流中的跨时间结构；traditional 方法在此处更适合作为非学习表示下界和计算成本参照。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.88\linewidth]{{mvsec_aee_float64.pdf}}
  \caption{{MVSEC optical flow 统一协议 AEE 对比，使用 2026-05-16 float64 timestamp 数据重跑结果。AEE 越低越好，横轴采用局部放大。}}
  \label{{fig:mvsec}}
\end{{figure}}

\paragraph{{MVSEC Updated Result.}} 最新 MVSEC float64 重跑后，整体最佳方法变为 {latex_escape(str(mvsec_best['method_label']))}，AEE 为 {float(mvsec_best['aee']):.4f}，outlier rate 为 {float(mvsec_best['outlier_percent']):.2f}\%。最佳 traditional 方法是 {latex_escape(str(mvsec_best_trad['method_label']))}，AEE 为 {float(mvsec_best_trad['aee']):.4f}；最佳 learning-based 方法是 {latex_escape(str(mvsec_best_learning['method_label']))}，AEE 为 {float(mvsec_best_learning['aee']):.4f}。与 5 月 8 日旧表不同，新表不再支持 ``traditional Timestamp Image 是 MVSEC 最优'' 的结论；在修正 float64 timestamp 与 timestamp-aligned protocol 后，EST 和 GET 均略优于最佳 traditional baseline。

\paragraph{{Takeaway For Presentation.}} 当前证据支持更细的分任务结论。N-MNIST 上两类方法都接近天花板，最佳 learning-based 只比最佳 traditional 高 0.32 个百分点，因此不应过度解读微小排名；N-Caltech101 上 learning-based 优势清晰，最佳方法比最佳 traditional 高 17.51 个百分点；MVSEC float64 重跑后，learning-based 的 EST/GET 在统一 decoder 下略优于 traditional Voxel Grid/Timestamp Image，但差距很小，说明传统事件表示仍然是强 baseline。汇报时应强调：传统方法不是弱对照，尤其在简单分类和 dense motion estimation 中很有竞争力；learning-based 方法的优势在更复杂分类任务中最明显，在本次 MVSEC 统一协议下体现为小幅领先。

\end{{document}}
"""
    (OUT / "traditional_vs_learning_float64_report_cn.tex").write_text(tex, encoding="utf-8")

    readme = f"""# Traditional vs Learning-based Comparison, 2026-05-16 Float64 Update

This artifact refreshes the cross-task comparison after the MVSEC optical-flow rerun with float64 event timestamps and timestamp-aligned event/flow windows.

## What Changed

- MVSEC now uses `optical-flow/results_float64_cached_20260516/summary_all/mvsec_e100_earlystop_summary.csv`.
- All 11 local MVSEC methods are included: 6 learning-based and 5 traditional.
- The old May 8 MVSEC conclusion is superseded. With corrected float64/timestamp alignment, EST is the best local MVSEC method.
- Classification tables are unchanged from the previously aligned N-MNIST and N-Caltech101 runs.

## Key MVSEC Results

| Group | Best method | AEE | Outlier % |
| --- | --- | ---: | ---: |
| Learning-based | {mvsec_best_learning['method_label']} | {float(mvsec_best_learning['aee']):.4f} | {float(mvsec_best_learning['outlier_percent']):.2f} |
| Traditional | {mvsec_best_trad['method_label']} | {float(mvsec_best_trad['aee']):.4f} | {float(mvsec_best_trad['outlier_percent']):.2f} |
| Overall | {mvsec_best['method_label']} | {float(mvsec_best['aee']):.4f} | {float(mvsec_best['outlier_percent']):.2f} |

## Files

- `classification_full_aligned_summary.csv`: existing aligned classification comparison.
- `mvsec_comparison_summary.csv`: latest float64 MVSEC optical-flow comparison.
- `traditional_vs_learning_float64_report_cn.tex`: Chinese report source.
- `traditional_vs_learning_float64_report_cn.pdf`: compiled report.
- `*_float64.png/pdf`: regenerated MVSEC figure.
- `nmnist_*` and `ncaltech101_*`: carried forward for the full cross-task report.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trad_rows = load_traditional_rows()
    nmnist_learning = load_nmnist_learning_rows()
    ncal_learning = load_ncal_learning_rows()
    cls_rows = sorted(
        trad_rows + nmnist_learning + ncal_learning,
        key=lambda r: (str(r["dataset"]), str(r["group"]), -float(r["accuracy_pct"])),
    )
    mvsec_rows = load_mvsec_rows()

    write_csv(OUT / "classification_full_aligned_summary.csv", cls_rows)
    write_csv(OUT / "nmnist_learning_full_aligned_summary.csv", nmnist_learning)
    write_csv(OUT / "mvsec_comparison_summary.csv", mvsec_rows)

    plot_accuracy(cls_rows, "nmnist", "nmnist_accuracy_full_aligned", "N-MNIST Classification: Full Aligned Results", xmin_pad=1.2)
    plot_nmnist_efficiency(cls_rows)
    plot_accuracy(cls_rows, "ncaltech101", "ncaltech101_accuracy_full_aligned", "N-Caltech101 Classification", xmin_pad=3.0)
    plot_mvsec(mvsec_rows)
    write_report(cls_rows, mvsec_rows)
    print(f"Wrote float64 comparison report assets to {OUT}")


if __name__ == "__main__":
    main()
