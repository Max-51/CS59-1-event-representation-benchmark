import argparse
import csv
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
METHODS = ["event_frame", "binary_event_image", "timestamp_image", "time_surface", "voxel_grid"]
METHOD_LABELS = {
    "event_frame": "Event Frame",
    "binary_event_image": "Binary Event Image",
    "timestamp_image": "Timestamp Image",
    "time_surface": "Time Surface",
    "voxel_grid": "Voxel Grid",
}


def read_json_from_tar(tar, path):
    member = tar.getmember(path)
    with tar.extractfile(member) as handle:
        return json.loads(handle.read().decode("utf-8"))


def read_text_lines_from_tar(tar, path):
    member = tar.getmember(path)
    with tar.extractfile(member) as handle:
        return handle.read().decode("utf-8").splitlines()


def collect_dataset(dataset, archive, prefix):
    rows = []
    history_rows = []
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        for method in METHODS:
            base = f"{prefix}/{dataset}/{method}"
            metrics_path = f"{base}/metrics.json"
            history_path = f"{base}/history.jsonl"
            config_path = f"{base}/config.json"
            if metrics_path not in names:
                raise FileNotFoundError(metrics_path)

            metrics = read_json_from_tar(tar, metrics_path)
            config = read_json_from_tar(tar, config_path) if config_path in names else {}
            training = metrics.get("training", config.get("training", {}))
            test = metrics.get("test", {})
            test_rep = test.get("representation", {})

            rows.append(
                {
                    "dataset": dataset,
                    "task": "classification",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "type": "Traditional",
                    "best_epoch": metrics.get("best_epoch"),
                    "best_val_accuracy": metrics.get("best_val_accuracy"),
                    "test_accuracy": test.get("accuracy"),
                    "test_loss": test.get("loss"),
                    "total_seconds": metrics.get("total_seconds"),
                    "stopped_early": metrics.get("stopped_early", False),
                    "epochs": training.get("epochs"),
                    "batch_size": training.get("batch_size"),
                    "device": training.get("device"),
                    "input_channels": metrics.get("representation", {}).get("input_channels"),
                    "test_samples": test.get("samples"),
                    "mean_events_test": test_rep.get("mean_events"),
                    "mean_build_seconds_test": test_rep.get("mean_build_seconds"),
                    "shape_counts_test": json.dumps(test_rep.get("shape_counts", {}), sort_keys=True),
                    "records_archive": archive.name,
                }
            )

            if history_path in names:
                for line in read_text_lines_from_tar(tar, history_path):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    history_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "epoch": item.get("epoch"),
                            "train_loss": item.get("train", {}).get("loss"),
                            "train_accuracy": item.get("train", {}).get("accuracy"),
                            "val_loss": item.get("val", {}).get("loss"),
                            "val_accuracy": item.get("val", {}).get("accuracy"),
                            "best_val_accuracy": item.get("best_val_accuracy"),
                            "no_improve": item.get("no_improve"),
                        }
                    )
    return rows, history_rows


def save_accuracy_bar(df, dataset, output_dir):
    subset = df[df["dataset"] == dataset].copy()
    subset = subset.sort_values("test_accuracy", ascending=False)
    colors = ["#28666e", "#7c9885", "#f2cc8f", "#f4a261", "#bc6c25"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(subset["method_label"], subset["test_accuracy"], color=colors[: len(subset)])
    ax.set_title(f"{dataset.upper()} Traditional Baselines: Test Accuracy")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(max(0, subset["test_accuracy"].min() - 0.05), min(1.0, subset["test_accuracy"].max() + 0.04))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / f"{dataset}_traditional_accuracy_bar.png", dpi=180)
    plt.close(fig)


def save_accuracy_time_scatter(df, dataset, output_dir):
    subset = df[df["dataset"] == dataset].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(subset["total_seconds"] / 60.0, subset["test_accuracy"], s=95, color="#28666e")
    for _, row in subset.iterrows():
        ax.annotate(row["method_label"], (row["total_seconds"] / 60.0, row["test_accuracy"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_title(f"{dataset.upper()} Accuracy vs Training Time")
    ax.set_xlabel("Training time (minutes)")
    ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / f"{dataset}_traditional_accuracy_time_scatter.png", dpi=180)
    plt.close(fig)


def save_training_curve(history, dataset, output_dir):
    subset = history[history["dataset"] == dataset].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method, group in subset.groupby("method", sort=False):
        ax.plot(group["epoch"], group["val_accuracy"], marker="o", linewidth=1.5, markersize=3, label=METHOD_LABELS[method])
    ax.set_title(f"{dataset.upper()} Validation Accuracy Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / f"{dataset}_traditional_val_accuracy_curves.png", dpi=180)
    plt.close(fig)


def save_learning_based_template(output_dir):
    path = output_dir / "tables" / "learning_based_classification_template.csv"
    fieldnames = [
        "dataset",
        "task",
        "method",
        "method_label",
        "type",
        "best_epoch",
        "best_val_accuracy",
        "test_accuracy",
        "test_loss",
        "total_seconds",
        "stopped_early",
        "epochs",
        "batch_size",
        "device",
        "downstream_model",
        "split",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "nmnist",
                "task": "classification",
                "method": "example_learning_method",
                "method_label": "Example Learning Method",
                "type": "Learning-based",
                "best_epoch": "",
                "best_val_accuracy": "",
                "test_accuracy": "",
                "test_loss": "",
                "total_seconds": "",
                "stopped_early": "",
                "epochs": "",
                "batch_size": "",
                "device": "cuda",
                "downstream_model": "ResNet18",
                "split": "official train/test, train split validation",
                "notes": "Replace this row with benchmark-group results.",
            }
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Build traditional classification result tables and figures.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing nmnist_traditional_records.tar.gz and ncaltech101_traditional_records.tar.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory where tables/ and figures/ will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    archives = [
        (
            "nmnist",
            input_dir / "nmnist_traditional_records.tar.gz",
            "mnt/outputs/traditional/classification",
        ),
        (
            "ncaltech101",
            input_dir / "ncaltech101_traditional_records.tar.gz",
            "traditional/classification",
        ),
    ]

    rows = []
    history_rows = []
    for dataset, archive, prefix in archives:
        dataset_rows, dataset_history = collect_dataset(dataset, archive, prefix)
        rows.extend(dataset_rows)
        history_rows.extend(dataset_history)

    df = pd.DataFrame(rows)
    history = pd.DataFrame(history_rows)
    df.to_csv(output_dir / "tables" / "traditional_classification_results.csv", index=False)
    history.to_csv(output_dir / "tables" / "traditional_classification_history.csv", index=False)
    save_learning_based_template(output_dir)

    for dataset in ["nmnist", "ncaltech101"]:
        save_accuracy_bar(df, dataset, output_dir)
        save_accuracy_time_scatter(df, dataset, output_dir)
        save_training_curve(history, dataset, output_dir)

    summary = df[
        [
            "dataset",
            "method",
            "best_epoch",
            "best_val_accuracy",
            "test_accuracy",
            "test_loss",
            "total_seconds",
            "stopped_early",
        ]
    ].sort_values(["dataset", "test_accuracy"], ascending=[True, False])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
