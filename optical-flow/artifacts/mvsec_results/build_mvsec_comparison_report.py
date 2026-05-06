from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib_cache_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "binary_event_image": "Binary Event Image",
    "event_frame": "Event Frame",
    "time_surface": "Time Surface",
    "timestamp_image": "Timestamp Image",
    "voxel_grid": "Voxel Grid",
    "ergo": "ERGO",
    "est": "EST",
    "event_pretraining": "Event Pre-training",
    "evrepsl": "EvRepSL",
    "get": "GET",
    "matrixlstm": "MatrixLSTM",
}


TRADITIONAL_METHODS = {
    "binary_event_image",
    "event_frame",
    "time_surface",
    "timestamp_image",
    "voxel_grid",
}


def extract_archive(path: Path, target: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(target)


def find_json_results(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if "/results/" in str(path))


def find_curve(root: Path, method: str) -> Path | None:
    candidates = sorted(root.rglob(f"*{method}*curve.csv"))
    return candidates[0] if candidates else None


def read_result(path: Path, group: str, source_archive: Path, extracted_root: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    method = str(data["adapter_name"])
    curve_path = find_curve(extracted_root, method)
    return {
        "dataset": "mvsec",
        "task": "optical_flow",
        "group": group,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "aee": data.get("aee"),
        "outlier_percent": data.get("outlier_percent"),
        "best_epoch": data.get("best_epoch"),
        "best_val_aee": data.get("best_val_aee"),
        "epochs_completed": data.get("epochs_completed"),
        "early_stopped": data.get("early_stopped"),
        "train_windows": data.get("train_windows"),
        "eval_windows": data.get("eval_windows"),
        "channels": data.get("channels"),
        "window_size": data.get("window_size"),
        "stride": data.get("stride"),
        "early_stop_patience": data.get("early_stop_patience"),
        "early_stop_min_delta": data.get("early_stop_min_delta"),
        "early_stop_val_windows": data.get("early_stop_val_windows_requested"),
        "early_stop_val_strategy": data.get("early_stop_val_strategy_requested"),
        "source_archive": source_archive.name,
        "source_json": str(path.relative_to(extracted_root)),
        "source_curve": str(curve_path.relative_to(extracted_root)) if curve_path else "",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_curve_rows(curve_path: Path, group: str, method: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with curve_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["group"] = group
            row["method"] = method
            row["method_label"] = METHOD_LABELS.get(method, method)
            rows.append(row)
    return rows


def load_curves(records: list[dict[str, object]], traditional_root: Path, learning_root: Path) -> list[dict[str, object]]:
    curves: list[dict[str, object]] = []
    for record in records:
        source_curve = str(record["source_curve"])
        if not source_curve:
            continue
        root = traditional_root if record["group"] == "Traditional" else learning_root
        curve_path = root / source_curve
        curves.extend(read_curve_rows(curve_path, str(record["group"]), str(record["method"])))
    return curves


def plot_bar(records: list[dict[str, object]], key: str, ylabel: str, path: Path) -> None:
    sorted_records = sorted(records, key=lambda row: float(row[key]))
    labels = [str(row["method_label"]) for row in sorted_records]
    values = [float(row[key]) for row in sorted_records]
    colors = ["#4C78A8" if row["group"] == "Learning-based" else "#F58518" for row in sorted_records]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("MVSEC Original Protocol: Traditional vs Learning-based")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#F58518", label="Traditional"),
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="Learning-based"),
    ]
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_aee_vs_outlier(records: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for row in records:
        color = "#4C78A8" if row["group"] == "Learning-based" else "#F58518"
        marker = "o" if row["group"] == "Learning-based" else "s"
        ax.scatter(float(row["aee"]), float(row["outlier_percent"]), c=color, marker=marker, s=75)
        ax.text(float(row["aee"]) + 0.004, float(row["outlier_percent"]) + 0.03, str(row["method_label"]), fontsize=8)
    ax.set_xlabel("AEE (lower is better)")
    ax.set_ylabel("Outlier % (lower is better)")
    ax.set_title("MVSEC Accuracy Trade-off")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_curves(curves: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for method in sorted({str(row["method"]) for row in curves}):
        method_rows = [row for row in curves if row["method"] == method]
        method_rows.sort(key=lambda row: int(row["epoch"]))
        group = method_rows[0]["group"]
        color = "#4C78A8" if group == "Learning-based" else "#F58518"
        linestyle = "-" if group == "Learning-based" else "--"
        ax.plot(
            [int(row["epoch"]) for row in method_rows],
            [float(row["val_aee"]) for row in method_rows],
            label=METHOD_LABELS.get(method, method),
            color=color,
            linestyle=linestyle,
            alpha=0.75,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AEE (lower is better)")
    ax.set_title("MVSEC Early-stop Validation Curves")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_readme(path: Path, records: list[dict[str, object]], traditional_archive: Path, learning_archive: Path) -> None:
    records_by_aee = sorted(records, key=lambda row: float(row["aee"]))
    best = records_by_aee[0]
    best_traditional = min((row for row in records if row["group"] == "Traditional"), key=lambda row: float(row["aee"]))
    best_learning = min((row for row in records if row["group"] == "Learning-based"), key=lambda row: float(row["aee"]))

    lines = [
        "# MVSEC Traditional vs Learning-based Result Package, 2026-05-06",
        "",
        "This folder records a lightweight comparison artifact for the MVSEC optical-flow task.",
        "It contains only derived tables and figures. It does not contain raw MVSEC HDF5/NPZ data",
        "or model checkpoints.",
        "",
        "## Source Archives",
        "",
        f"- traditional: `{traditional_archive.name}`",
        f"- learning-based: `{learning_archive.name}`",
        "",
        "## Protocol",
        "",
        "- train: `outdoor_day1 + outdoor_day2`",
        "- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`",
        "- events: 6M extracted left-camera events per sequence",
        "- flow GT: generated full `*_gt_flow_full.npz`",
        "- window size / stride: 200 / 200",
        "- decoder: shared `EVFlowNetLike`",
        "- max epochs: 100",
        "- early stop: patience 10, min delta 0.001",
        "- early-stop validation: 1000 block-random outdoor windows",
        "- metrics: AEE and outlier percent, lower is better",
        "",
        "## Result Summary",
        "",
        "| Group | Method | AEE | Outlier % | Best epoch | Channels |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in records_by_aee:
        lines.append(
            f"| {row['group']} | {row['method_label']} | {float(row['aee']):.6f} | "
            f"{float(row['outlier_percent']):.6f} | {row['best_epoch']} | {row['channels']} |"
        )
    lines.extend(
        [
            "",
            "## Quick Interpretation",
            "",
            f"- Best overall AEE in this unified run: `{best['method_label']}` ({best['group']}), AEE {float(best['aee']):.6f}.",
            f"- Best traditional method: `{best_traditional['method_label']}`, AEE {float(best_traditional['aee']):.6f}.",
            f"- Best learning-based method: `{best_learning['method_label']}`, AEE {float(best_learning['aee']):.6f}.",
            "- The comparison is protocol-aligned: all methods use the same train/eval split,",
            "  shared decoder, early-stop rule, and metrics. It is a unified benchmark comparison,",
            "  not a paper-identical rerun of each method's original private training stack.",
            "",
            "## Files",
            "",
            "- `tables/mvsec_comparison_results.csv`: one row per method.",
            "- `tables/mvsec_comparison_curves.csv`: epoch-level validation curves.",
            "- `figures/mvsec_aee_bar.png`: AEE comparison.",
            "- `figures/mvsec_outlier_bar.png`: outlier-percent comparison.",
            "- `figures/mvsec_aee_outlier_scatter.png`: AEE/outlier trade-off.",
            "- `figures/mvsec_val_aee_curves.png`: early-stop validation curves.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MVSEC traditional vs learning-based comparison artifacts.")
    parser.add_argument("--traditional-archive", type=Path, required=True)
    parser.add_argument("--learning-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mvsec_compare_") as tmp:
        tmp_root = Path(tmp)
        traditional_root = tmp_root / "traditional"
        learning_root = tmp_root / "learning"
        traditional_root.mkdir()
        learning_root.mkdir()
        extract_archive(args.traditional_archive, traditional_root)
        extract_archive(args.learning_archive, learning_root)

        records: list[dict[str, object]] = []
        for path in find_json_results(traditional_root):
            records.append(read_result(path, "Traditional", args.traditional_archive, traditional_root))
        for path in find_json_results(learning_root):
            records.append(read_result(path, "Learning-based", args.learning_archive, learning_root))

        protocol_keys = [
            "train_windows",
            "eval_windows",
            "window_size",
            "stride",
            "early_stop_patience",
            "early_stop_min_delta",
            "early_stop_val_windows",
            "early_stop_val_strategy",
        ]
        for key in protocol_keys:
            values = {record[key] for record in records}
            if len(values) != 1:
                raise ValueError(f"Protocol mismatch for {key}: {values}")

        curves = load_curves(records, traditional_root, learning_root)

    write_csv(output_dir / "tables" / "mvsec_comparison_results.csv", sorted(records, key=lambda row: float(row["aee"])))
    write_csv(output_dir / "tables" / "mvsec_comparison_curves.csv", curves)
    plot_bar(records, "aee", "AEE (lower is better)", output_dir / "figures" / "mvsec_aee_bar.png")
    plot_bar(records, "outlier_percent", "Outlier % (lower is better)", output_dir / "figures" / "mvsec_outlier_bar.png")
    plot_aee_vs_outlier(records, output_dir / "figures" / "mvsec_aee_outlier_scatter.png")
    plot_curves(curves, output_dir / "figures" / "mvsec_val_aee_curves.png")
    write_readme(output_dir / "README.md", records, args.traditional_archive, args.learning_archive)

    print(f"Wrote MVSEC comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
