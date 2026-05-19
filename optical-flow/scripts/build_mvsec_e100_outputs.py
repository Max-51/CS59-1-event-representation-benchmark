from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LEARNING_METHOD_ORDER = ["ergo", "est", "event_pretraining", "evrepsl", "get", "matrixlstm"]
TRADITIONAL_METHOD_ORDER = ["event_frame", "binary_event_image", "timestamp_image", "time_surface", "voxel_grid"]
METHOD_LABELS = {
    "ergo": "ERGO",
    "est": "EST",
    "event_pretraining": "Event Pre-training",
    "evrepsl": "EvRepSL",
    "get": "GET",
    "matrixlstm": "MatrixLSTM",
    "event_frame": "Event Frame",
    "binary_event_image": "Binary Event Image",
    "timestamp_image": "Timestamp Image",
    "time_surface": "Time Surface",
    "voxel_grid": "Voxel Grid",
}


def _load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    method = str(data["adapter_name"])
    return {
        "method": method,
        "aee": float(data["aee"]),
        "outlier_percent": float(data["outlier_percent"]),
        "non_outlier_percent": 100.0 - float(data["outlier_percent"]),
        "epochs_completed": int(data.get("epochs_completed") or 0),
        "best_epoch": int(data.get("best_epoch") or 0),
        "best_val_aee": float(data.get("best_val_aee") or 0.0),
        "early_stopped": bool(data.get("early_stopped")),
        "train_windows": int(data.get("train_windows") or 0),
        "eval_windows": int(data.get("eval_windows") or 0),
        "valid_count": int(data.get("valid_count") or 0),
        "window_alignment": str(data.get("window_alignment") or ""),
        "metric_scope": str(data.get("metric_scope") or "full_gt_valid"),
        "file": str(path.as_posix()),
    }


def _find_result_files(results_dir: Path) -> list[Path]:
    files = sorted(results_dir.glob("original_od12_if123_full6m_*_e100_block-random_earlystop.json"))
    if not files:
        files = sorted(results_dir.glob("*_earlystop.json"))
    return files


def _write_summary_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "aee",
        "outlier_percent",
        "non_outlier_percent",
        "epochs_completed",
        "best_epoch",
        "best_val_aee",
        "early_stopped",
        "train_windows",
        "eval_windows",
        "valid_count",
        "window_alignment",
        "metric_scope",
        "file",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# MVSEC Optical Flow E100 Early-Stop Summary",
        "",
        "Protocol: train on `outdoor_day1 + outdoor_day2`, evaluate on `indoor_flying1/2/3`, event input from extracted left-camera events, full generated GT flow frames, max 100 epochs, early-stop patience 10, block-random validation from outdoor train set, timestamp-aligned event/flow windows.",
        "",
        "Metrics are AEE and KITTI-style outlier percentage over valid GT flow pixels. Lower is better.",
        "",
        "| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE | Train windows | Eval windows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {aee:.4f} | {outlier_percent:.2f} | {non_outlier_percent:.2f} | {epochs_completed} | {best_epoch} | {best_val_aee:.4f} | {train_windows} | {eval_windows} |".format(
                **row,
            )
        )
    include_omni = any(row["method"] in LEARNING_METHOD_ORDER for row in rows)
    if include_omni:
        lines.extend(
            [
                "| OmniEvent* | 0.9900 | 3.24 | 96.76 | paper | paper | paper | paper | paper |",
                "",
                "*OmniEvent is a paper-reported reference row. Its displayed values are simple averages from the OmniEvent paper's MVSEC `indoor_flying1/2/3` results, not a local run in this repository. Source: arXiv:2508.01842, Table 2.",
                "",
            ]
        )
    notes = [
        "## Reporting Notes",
        "",
        "- This is a unified downstream optical-flow benchmark / adapted reproduction, not an exact reimplementation of each paper's original optical-flow decoder.",
        "- Local methods are compared under the same EVFlowNet-like decoder and the same MVSEC train/eval protocol, so the comparison is fair inside this benchmark.",
    ]
    if include_omni:
        notes.append("- `OmniEvent*` is included only as reported-only context and should not be ranked as a local result from this pipeline.")
    notes.extend(
        [
            "- Local CSV curves and SVG figures are generated from the run JSON and curve CSV files.",
            "",
        ]
    )
    lines.extend(notes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _plot_bars(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [METHOD_LABELS.get(row["method"], row["method"]) for row in rows]
    colors = ["#2a6f97" if row["method"] == "est" else "#8fb3c7" for row in rows]

    for key, ylabel, filename in [
        ("aee", "AEE (lower is better)", "mvsec_e100_earlystop_aee.svg"),
        ("outlier_percent", "Outlier % (lower is better)", "mvsec_e100_earlystop_outlier.svg"),
    ]:
        values = [row[key] for row in rows]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(f"MVSEC optical flow {ylabel}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / filename)
        plt.close(fig)


def _load_curve(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "epoch": float(row["epoch"]),
                    "train_loss": float(row["train_loss"]) if row.get("train_loss") else float("nan"),
                    "val_aee": float(row["val_aee"]) if row.get("val_aee") else float("nan"),
                }
            )
        return rows


def _plot_curves(curve_dir: Path, figures_dir: Path, method_order: list[str]) -> None:
    import matplotlib.pyplot as plt

    curve_files = {
        method: curve_dir / f"original_od12_if123_full6m_{method}_e100_block-random_curve.csv"
        for method in method_order
    }
    existing = {method: path for method, path in curve_files.items() if path.exists()}
    if not existing:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    for key, ylabel, filename in [
        ("train_loss", "Train loss", "mvsec_e100_earlystop_train_loss_curve.svg"),
        ("val_aee", "Validation AEE", "mvsec_e100_earlystop_val_curve.svg"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for method, path in existing.items():
            curve = _load_curve(path)
            ax.plot(
                [row["epoch"] for row in curve],
                [row[key] for row in curve],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"MVSEC optical flow {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / filename)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MVSEC E100 summary tables and figures from run outputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--curve-dir", type=Path, default=Path("logs/curves"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--method-group",
        choices=["learning", "traditional", "all", "auto"],
        default="learning",
        help="Which local methods to require and summarize.",
    )
    args = parser.parse_args()

    files = _find_result_files(args.results_dir)
    rows_by_method = {_load_result(path)["method"]: _load_result(path) for path in files}
    if args.method_group == "learning":
        method_order = LEARNING_METHOD_ORDER
    elif args.method_group == "traditional":
        method_order = TRADITIONAL_METHOD_ORDER
    elif args.method_group == "all":
        method_order = LEARNING_METHOD_ORDER + TRADITIONAL_METHOD_ORDER
    else:
        known_order = LEARNING_METHOD_ORDER + TRADITIONAL_METHOD_ORDER
        method_order = [method for method in known_order if method in rows_by_method]
        method_order.extend(sorted(method for method in rows_by_method if method not in method_order))

    missing = [method for method in method_order if method not in rows_by_method]
    if missing:
        raise SystemExit(f"Missing result JSON for methods: {', '.join(missing)}")
    rows = [rows_by_method[method] for method in method_order]

    _write_summary_csv(rows, args.summary_dir / "mvsec_e100_earlystop_summary.csv")
    _write_summary_md(rows, args.summary_dir / "mvsec_e100_earlystop_summary.md")
    _plot_bars(rows, args.figures_dir)
    _plot_curves(args.curve_dir, args.figures_dir, method_order)

    print(f"wrote {args.summary_dir / 'mvsec_e100_earlystop_summary.csv'}")
    print(f"wrote {args.summary_dir / 'mvsec_e100_earlystop_summary.md'}")
    print(f"wrote figures under {args.figures_dir}")


if __name__ == "__main__":
    main()
