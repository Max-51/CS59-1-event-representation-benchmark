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
OUT = ROOT / "artifacts" / "traditional_baseline_analysis" / "20260508_update"
TRAD_CLS = ROOT / "artifacts" / "traditional_classification" / "tables" / "traditional_classification_results.csv"
LEARN_CLS_ROOT = ROOT / "results" / "classification"
MVSEC = ROOT / "optical-flow" / "artifacts" / "mvsec_results" / "20260506_traditional_vs_learning" / "tables" / "mvsec_comparison_results.csv"

NMNIST_LOCAL_RESULTS = [
    ROOT / "outputs/learning_classification/nmnist/20260508_cpu/results/est/est_nmnist.json",
    ROOT / "outputs/learning_classification/nmnist/20260508_cpu/results/ergo/ergo_nmnist.json",
    ROOT / "outputs/learning_classification/nmnist/20260508_cpu/results/event_pretraining/event_pretraining_nmnist.json",
    ROOT / "outputs/learning_classification/nmnist/20260508_cpu_extra/results/evrepsl/evrepsl_nmnist.json",
    ROOT / "outputs/learning_classification/nmnist/20260508_cpu_extra/results/get/get_nmnist.json",
    ROOT / "outputs/learning_classification/nmnist/20260508_cpu_extra/results/omnievent/omnievent_nmnist.json",
    ROOT / "outputs/learning_classification/nmnist/20260508_cpu_matrixlstm/results/matrix_lstm/matrix_lstm_nmnist.json",
]

METHOD_LABELS = {
    "est": "EST",
    "ergo": "ERGO",
    "evrepsl": "EvRepSL",
    "event_pretraining": "Event Pre-training",
    "matrix_lstm": "Matrix-LSTM",
    "get": "GET",
    "omnievent": "OmniEvent",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_full_classification_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(TRAD_CLS):
        rows.append(
            {
                "dataset": row["dataset"],
                "group": "Traditional",
                "method": row["method_label"],
                "accuracy_pct": 100.0 * float(row["test_accuracy"]),
                "protocol": "full official/test split",
                "source": row["records_archive"],
                "note": "Traditional representation + ResNet18 classifier",
            }
        )

    for path in sorted(LEARN_CLS_ROOT.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset_raw = str(data.get("dataset", "")).lower()
        if "nmnist" in dataset_raw or "n-mnist" in dataset_raw:
            dataset = "nmnist"
        elif "ncaltech" in dataset_raw or "n-caltech" in dataset_raw:
            dataset = "ncaltech101"
        else:
            continue

        method = METHOD_LABELS.get(str(data.get("method", path.parent.name)), str(data.get("method", path.parent.name)))
        if path.name == "est_ncaltech101.json":
            method = "EST adaptation"
        elif path.name == "est_e2e_ncaltech101.json":
            method = "EST End-to-End"

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
                "protocol": "repository recorded full result",
                "source": str(path.relative_to(ROOT)),
                "note": "Learning-based classification result already recorded in repository",
            }
        )
    return sorted(rows, key=lambda r: (str(r["dataset"]), str(r["group"]), -float(r["accuracy_pct"])))


def load_nmnist_local_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in NMNIST_LOCAL_RESULTS:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        method_key = str(data["method"])
        history = data.get("train_history", [])
        first = history[0] if history else {}
        last = history[-1] if history else {}
        rows.append(
            {
                "dataset": "nmnist",
                "group": "Learning-based",
                "method": METHOD_LABELS.get(method_key, method_key),
                "best_test_accuracy_pct": float(data["best_test_accuracy_pct"]),
                "epoch1_train_accuracy_pct": 100.0 * float(first.get("train_acc", 0.0)),
                "final_train_loss": float(last.get("train_loss", 0.0)),
                "epochs": int(data["epochs"]),
                "actual_epochs": int(data.get("early_stopping", {}).get("actual_epochs", data["epochs"])),
                "batch_size": int(data["batch_size"]),
                "lr": float(data["lr"]),
                "weight_decay": float(data["weight_decay"]),
                "seed": int(data["seed"]),
                "max_events": 50000,
                "train_limit": 2048,
                "test_limit": 512,
                "device": str(data["gpu"]),
                "in_channels": int(data["in_channels"]),
                "source": str(path.relative_to(ROOT)),
                "note": "local sanity-scale run; not a full official-split benchmark",
            }
        )
    return sorted(rows, key=lambda r: (-float(r["epoch1_train_accuracy_pct"]), float(r["final_train_loss"])))


def style_axis(ax) -> None:
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#D0D0D0", alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_accuracy(dataset: str, rows: list[dict[str, object]], filename: str, title: str) -> None:
    subset = [r for r in rows if r["dataset"] == dataset]
    subset = sorted(subset, key=lambda r: float(r["accuracy_pct"]))
    labels = [str(r["method"]) for r in subset]
    values = [float(r["accuracy_pct"]) for r in subset]
    colors = ["#D55E00" if r["group"] == "Traditional" else "#0072B2" for r in subset]

    fig_h = max(4.2, 0.38 * len(subset) + 1.3)
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    y = np.arange(len(subset))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title(title)
    style_axis(ax)
    ax.set_xlim(max(0, min(values) - 3.0), min(100.4, max(values) + 1.8))
    for bar, value in zip(bars, values):
        ax.text(value + 0.10, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8.2)
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
    fig.savefig(OUT / f"{filename}.png", dpi=260)
    plt.close(fig)


def plot_nmnist_local(rows: list[dict[str, object]]) -> None:
    labels = [str(r["method"]) for r in rows]
    epoch1 = [float(r["epoch1_train_accuracy_pct"]) for r in rows]
    loss = [float(r["final_train_loss"]) for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bars = ax.barh(y, epoch1, color="#009E73", edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Epoch-1 train accuracy (%)")
    ax.set_title("N-MNIST Learning-based Sanity-scale Runs")
    ax.set_xlim(95.8, 97.4)
    style_axis(ax)
    for bar, value, final_loss in zip(bars, epoch1, loss):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f} / loss {final_loss:.4f}", va="center", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(OUT / "nmnist_learning_sanity_epoch1.pdf")
    fig.savefig(OUT / "nmnist_learning_sanity_epoch1.png", dpi=260)
    plt.close(fig)


def plot_mvsec(rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda r: float(r["aee"]), reverse=True)
    labels = [r["method_label"] for r in rows]
    values = [float(r["aee"]) for r in rows]
    colors = ["#D55E00" if r["group"] == "Traditional" else "#0072B2" for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AEE (lower is better)")
    ax.set_title("MVSEC Optical Flow: Unified Protocol")
    ax.set_xlim(2.74, 3.05)
    style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8.0)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#D55E00", label="Traditional"),
            plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="Learning-based"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUT / "mvsec_aee_updated.pdf")
    fig.savefig(OUT / "mvsec_aee_updated.png", dpi=260)
    plt.close(fig)


def tex_table(rows: list[dict[str, object]], dataset: str, max_rows: int | None = None) -> str:
    subset = [r for r in rows if r["dataset"] == dataset]
    subset = sorted(subset, key=lambda r: -float(r["accuracy_pct"]))
    if max_rows:
        subset = subset[:max_rows]
    lines = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Group & Method & Accuracy (\%) \\",
        r"\midrule",
    ]
    for r in subset:
        lines.append(f"{r['group']} & {latex_escape(str(r['method']))} & {float(r['accuracy_pct']):.2f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_report(full_rows: list[dict[str, object]], local_rows: list[dict[str, object]], mvsec_rows: list[dict[str, str]]) -> None:
    nmnist_full = [r for r in full_rows if r["dataset"] == "nmnist"]
    ncal = [r for r in full_rows if r["dataset"] == "ncaltech101"]
    mvsec_best = min(mvsec_rows, key=lambda r: float(r["aee"]))
    mvsec_best_learning = min([r for r in mvsec_rows if r["group"] == "Learning-based"], key=lambda r: float(r["aee"]))
    best_nmnist = max(nmnist_full, key=lambda r: float(r["accuracy_pct"]))
    best_nmnist_learning = max([r for r in nmnist_full if r["group"] == "Learning-based"], key=lambda r: float(r["accuracy_pct"]))
    best_ncal = max(ncal, key=lambda r: float(r["accuracy_pct"]))
    best_ncal_trad = max([r for r in ncal if r["group"] == "Traditional"], key=lambda r: float(r["accuracy_pct"]))
    local_first_best = max(local_rows, key=lambda r: float(r["epoch1_train_accuracy_pct"]))
    local_loss_best = min(local_rows, key=lambda r: float(r["final_train_loss"]))

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=0.82in]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{xeCJK}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{caption}}
\usepackage{{array}}

\setmainfont{{Times New Roman}}
\setsansfont{{Arial}}
\setCJKmainfont{{Songti SC}}
\setCJKsansfont{{STHeiti}}
\setlength{{\parskip}}{{0.62em}}
\setlength{{\parindent}}{{0pt}}

\title{{Traditional 与 Learning-based 事件表示更新对比报告}}
\author{{}}
\date{{2026-05-08}}

\begin{{document}}
\sloppy
\maketitle

\paragraph{{实验范围与读数原则。}} 本报告更新了 N-MNIST learning-based 结果，并保留 N-Caltech101 与 MVSEC 的既有对比。为了避免混淆，N-MNIST 被分成两类结果：第一类是完整官方测试集或仓库记录的 full-result，对应传统方法和仓库已有 EvRepSL；第二类是本次新跑的 7 个 learning-based sanity-scale 结果，设置为 CPU、10 epoch、train/test limit 2048/512、batch size 32、learning rate 0.0001。这组 sanity 结果可以证明代码路径、表示构建和分类器训练是通的，但不应被当作完整 official-split benchmark 结论。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.86\linewidth]{{nmnist_full_accuracy_updated.pdf}}
  \caption{{N-MNIST 完整结果准确率对比。橙色为 traditional baseline，蓝色为仓库已有 learning-based full result。}}
  \label{{fig:nmnist-full}}
\end{{figure}}

\paragraph{{N-MNIST Full Result Saturation.}} 在完整 N-MNIST 结果中，最佳 traditional 方法是 Voxel Grid，准确率为 {float(best_nmnist['accuracy_pct']):.2f}\%；仓库已有 learning-based EvRepSL 为 {float(best_nmnist_learning['accuracy_pct']):.2f}\%。二者差距只有 {abs(float(best_nmnist['accuracy_pct']) - float(best_nmnist_learning['accuracy_pct'])):.2f} 个百分点，Time Surface、Event Frame 和 Timestamp Image 也均处在 98\% 以上。因此，N-MNIST 更适合作为接口和训练流程验证任务，而不是强区分表示能力的核心任务。Binary Event Image 为 95.06\%，是完整 N-MNIST 中唯一明显掉队的方法，说明只保留二值触发信息会损失一部分时序和强度线索。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.86\linewidth]{{nmnist_learning_sanity_epoch1.pdf}}
  \caption{{新跑 N-MNIST learning-based sanity-scale 结果。所有方法在 2048/512 小规模划分上达到 100\% test accuracy，因此图中展示 epoch-1 train accuracy 和最终 train loss，用于观察收敛速度而非最终排名。}}
  \label{{fig:nmnist-sanity}}
\end{{figure}}

\paragraph{{N-MNIST Sanity Runs Are Not Ranking Evidence.}} 本次新增的 7 个 learning-based 本地结果在 2048/512 小规模 N-MNIST 设置上全部达到 100.00\% test accuracy，这说明这些方法的本地 adapter、表示张量和 ResNet18 classifier 路径均能正常工作。但由于测试样本只有 512 个，且 N-MNIST 本身接近饱和，这组结果不能用于声称某个 learning-based 方法显著优于另一个方法。若只看训练早期，EvRepSL 的 epoch-1 train accuracy 最高，为 {float(local_first_best['epoch1_train_accuracy_pct']):.2f}\%；最终 train loss 最低的也是 {latex_escape(str(local_loss_best['method']))}，为 {float(local_loss_best['final_train_loss']):.4f}。这更像是收敛速度差异，而不是泛化能力差异。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.92\linewidth]{{ncaltech101_accuracy_updated.pdf}}
  \caption{{N-Caltech101 分类准确率对比。蓝色为 learning-based 方法，橙色为 traditional baseline。}}
  \label{{fig:ncaltech}}
\end{{figure}}

\paragraph{{N-Caltech101 Separates Representation Capacity.}} 与 N-MNIST 不同，N-Caltech101 对表示能力的区分更明显。当前最佳 learning-based 方法是 {latex_escape(str(best_ncal['method']))}，准确率为 {float(best_ncal['accuracy_pct']):.2f}\%；最佳 traditional 方法是 {latex_escape(str(best_ncal_trad['method']))}，准确率为 {float(best_ncal_trad['accuracy_pct']):.2f}\%。二者相差 {float(best_ncal['accuracy_pct']) - float(best_ncal_trad['accuracy_pct']):.2f} 个百分点。这个差距说明，在类别更多、形状更复杂的事件分类任务上，学习型表示或端到端学习 pipeline 更容易组织跨时间的判别信息；traditional baseline 仍然重要，但更适合作为非学习表示下界。

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.88\linewidth]{{mvsec_aee_updated.pdf}}
  \caption{{MVSEC optical flow 统一协议下的 AEE 对比。AEE 越低越好，横轴采用局部放大以展示 2.79--3.02 区间内的差异。}}
  \label{{fig:mvsec}}
\end{{figure}}

\paragraph{{MVSEC Shows A Different Pattern.}} MVSEC optical flow 使用 outdoor\_day1/2 训练、indoor\_flying1/2/3 测试，并共享 EVFlowNet-like decoder，因此比分类任务更接近统一协议比较。在这个设置下，最佳方法是 {latex_escape(mvsec_best['method_label'])}，AEE 为 {float(mvsec_best['aee']):.4f}，outlier rate 为 {float(mvsec_best['outlier_percent']):.2f}\%；最佳 learning-based 方法是 {latex_escape(mvsec_best_learning['method_label'])}，AEE 为 {float(mvsec_best_learning['aee']):.4f}。这说明 timestamp-preserving traditional representation 在光流估计中具有很强竞争力。需要谨慎表述的是，这并不等价于 traditional 方法在原论文完整 setting 中全面超过 learning-based 方法，而是在本仓库统一 decoder 和统一 early-stopping protocol 下取得了更低 AEE。

\paragraph{{汇报建议。}} 晚上汇报可以采用三句话主线。第一，N-MNIST 基本接近饱和，完整结果中 Voxel Grid 与 EvRepSL 仅差 0.27 个百分点，新跑的 7 个 learning-based sanity 结果全部达到 100\%，因此它主要证明 pipeline 可运行。第二，N-Caltech101 才更能拉开差距，EST adaptation 比最佳 traditional Timestamp Image 高 17.51 个百分点，说明复杂分类任务上 learning-based 表示优势更明显。第三，MVSEC 呈现不同结论：统一协议下 Timestamp Image 的 AEE 低于当前 learning-based 最优 EST，说明传统事件表示并不是弱 baseline，尤其保留时间戳结构的表示值得在 survey 中重点讨论。

\end{{document}}
"""
    (OUT / "updated_comparison_report_cn.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full_rows = load_full_classification_rows()
    local_rows = load_nmnist_local_rows()
    mvsec_rows = read_csv(MVSEC)

    write_csv(OUT / "classification_full_comparison_summary.csv", full_rows)
    write_csv(OUT / "nmnist_learning_sanity_summary.csv", local_rows)
    write_csv(OUT / "mvsec_comparison_summary.csv", mvsec_rows)

    plot_accuracy("nmnist", full_rows, "nmnist_full_accuracy_updated", "N-MNIST Classification: Full Results")
    plot_nmnist_local(local_rows)
    plot_accuracy("ncaltech101", full_rows, "ncaltech101_accuracy_updated", "N-Caltech101 Classification")
    plot_mvsec(mvsec_rows)
    write_report(full_rows, local_rows, mvsec_rows)
    print(f"Wrote updated report assets to {OUT}")


if __name__ == "__main__":
    main()
