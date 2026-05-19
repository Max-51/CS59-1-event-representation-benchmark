import argparse
import json
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.datasets.gen1_detection import build_gen1_window_index, write_gen1_window_index
    from src.detection.gen1_benchmark import (
        Gen1YoloV6Dataset,
        append_jsonl,
        build_yolov6_model,
        checkpoint_payload,
        create_dataloader,
        create_optimizer,
        create_scheduler,
        evaluate_detection,
        load_checkpoint,
        progress_payload,
        save_checkpoint,
        save_json,
        train_one_epoch,
    )
except ModuleNotFoundError as exc:
    if "-h" in sys.argv or "--help" in sys.argv:
        build_gen1_window_index = None
        write_gen1_window_index = None
        Gen1YoloV6Dataset = None
        append_jsonl = None
        build_yolov6_model = None
        checkpoint_payload = None
        create_dataloader = None
        create_optimizer = None
        create_scheduler = None
        evaluate_detection = None
        load_checkpoint = None
        progress_payload = None
        save_checkpoint = None
        save_json = None
        train_one_epoch = None
    else:
        raise exc


METHODS = [
    "est",
    "ergo",
    "evrepsl",
    "get",
    "event_pretraining",
    "matrix_lstm",
    "event_frame",
    "event_count",
    "binary_event_image",
    "timestamp_image",
    "time_surface",
    "voxel_grid",
]


def ensure_indices(root, metadata_dir, window_us):
    root = Path(root).resolve()
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_paths = {}
    for split in ("train", "val", "test"):
        index_path = metadata_dir / f"{split}.jsonl"
        if not index_path.exists():
            entries = build_gen1_window_index(root, split, window_us=window_us)
            write_gen1_window_index(entries, index_path, root=root)
        index_paths[split] = index_path.resolve()
    return index_paths


def _metric_value(metrics, metric_name):
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(f"Metric '{metric_name}' was not produced by evaluation")
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv6 on indexed GEN1 windows for one event representation.")
    parser.add_argument("--root", required=True, help="GEN1 extracted dataset root")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--metadata-dir", default="metadata/gen1_windows")
    parser.add_argument("--config", default="tasks/detection/configs/yolov6n_gen1.py")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window-us", type=int, default=50_000)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=15,
        help="Stop after this many epochs without validation improvement. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stop-metric",
        default="map50_95",
        choices=["map50", "map50_95"],
        help="Validation metric used for best checkpoint and early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation metric improvement required to reset early stopping.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = Path(args.root).resolve()
    metadata_dir = Path(args.metadata_dir)
    config_path = Path(args.config)
    device = torch.device(args.device)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("artifacts") / "detection" / "gen1" / "default_run" / args.method
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / "progress.json"
    metrics_file = output_dir / "metrics.json"
    history_file = output_dir / "history.jsonl"
    checkpoints_dir = output_dir / "checkpoints"
    last_ckpt = checkpoints_dir / "last.pt"
    best_ckpt = checkpoints_dir / "best.pt"

    index_paths = ensure_indices(root, metadata_dir, args.window_us)
    representation_config = {"device": str(device), "return_numpy": False}
    datasets = {
        "train": Gen1YoloV6Dataset(
            index_paths["train"],
            args.method,
            root=root,
            representation_config=representation_config,
            img_size=args.img_size,
            max_windows=args.train_limit,
        ),
        "val": Gen1YoloV6Dataset(
            index_paths["val"],
            args.method,
            root=root,
            representation_config=representation_config,
            img_size=args.img_size,
            max_windows=args.val_limit,
        ),
        "test": Gen1YoloV6Dataset(
            index_paths["test"],
            args.method,
            root=root,
            representation_config=representation_config,
            img_size=args.img_size,
            max_windows=args.test_limit,
        ),
    }
    dataloaders = {
        "train": create_dataloader(datasets["train"], args.batch_size, True, args.num_workers),
        "val": create_dataloader(datasets["val"], args.batch_size, False, args.num_workers),
        "test": create_dataloader(datasets["test"], args.batch_size, False, args.num_workers),
    }

    _, model, criterion = build_yolov6_model(config_path, device, args.img_size)
    optimizer = create_optimizer(model, args.lr, args.momentum, args.weight_decay)
    scheduler = create_scheduler(optimizer, args.epochs)
    started_at = time.time()
    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0

    if args.resume and last_ckpt.exists():
        resume_payload = load_checkpoint(last_ckpt, model, optimizer, scheduler, map_location=device)
        start_epoch = int(resume_payload["epoch"]) + 1
        best_metric = float(
            resume_payload.get(
                "best_metric",
                resume_payload.get("best_map50", best_metric),
            )
        )
        best_epoch = resume_payload.get("best_epoch")
        epochs_without_improvement = int(resume_payload.get("epochs_without_improvement", 0))

    run_summary = {
        "method": args.method,
        "root": str(root),
        "metadata_dir": str(metadata_dir.resolve()),
        "config": str(config_path.resolve()),
        "device": str(device),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "datasets": {split: len(dataset) for split, dataset in datasets.items()},
        "train": [],
        "val": [],
        "test": None,
        "best_epoch": best_epoch,
        "best_metric_name": args.early_stop_metric,
        "best_metric": None if best_metric == float("-inf") else best_metric,
        "best_map50": None,
        "best_map50_95": None,
        "early_stop_patience": args.early_stop_patience,
        "stopped_early": False,
        "completed_epochs": start_epoch,
    }

    initial_payload = progress_payload(args.method, start_epoch, args.epochs, started_at=started_at, stage="initializing")
    save_json(progress_file, initial_payload)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model,
            criterion,
            optimizer,
            dataloaders["train"],
            device,
            epoch,
            args.img_size,
        )
        val_metrics = evaluate_detection(model, criterion, dataloaders["val"], device, args.img_size)
        scheduler.step()

        run_summary["train"].append(train_metrics)
        run_summary["val"].append(val_metrics)
        current_metric = _metric_value(val_metrics, args.early_stop_metric)
        improved = current_metric > best_metric + args.min_delta

        progress = progress_payload(
            args.method,
            epoch + 1,
            args.epochs,
            split_metrics={"val": val_metrics},
            train_metrics=train_metrics,
            started_at=started_at,
            stage="training",
        )
        save_json(progress_file, progress)
        append_jsonl(history_file, progress)

        args_dict = vars(args).copy()
        ckpt = checkpoint_payload(epoch, model, optimizer, scheduler, best_metric, args_dict)
        ckpt["best_metric_name"] = args.early_stop_metric

        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            run_summary["best_epoch"] = epoch + 1
            run_summary["best_metric"] = best_metric
            run_summary["best_map50"] = val_metrics["map50"]
            run_summary["best_map50_95"] = val_metrics["map50_95"]
            ckpt["best_map50"] = val_metrics["map50"]
            ckpt["best_map50_95"] = val_metrics["map50_95"]
            ckpt["best_metric"] = best_metric
            ckpt["best_epoch"] = best_epoch
            ckpt["epochs_without_improvement"] = epochs_without_improvement
            save_checkpoint(best_ckpt, ckpt)
        else:
            epochs_without_improvement += 1
            run_summary["best_epoch"] = best_epoch
            run_summary["best_metric"] = None if best_metric == float("-inf") else best_metric

        run_summary["completed_epochs"] = epoch + 1
        ckpt["best_metric"] = best_metric
        ckpt["best_epoch"] = best_epoch
        ckpt["best_map50"] = run_summary["best_map50"]
        ckpt["best_map50_95"] = run_summary["best_map50_95"]
        ckpt["epochs_without_improvement"] = epochs_without_improvement
        save_checkpoint(last_ckpt, ckpt)

        metrics_snapshot = run_summary | {"updated_at": time.time()}
        save_json(metrics_file, metrics_snapshot)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_metrics['mean_total_loss']:.4f} "
            f"val_map50={val_metrics['map50']:.4f} "
            f"val_map50_95={val_metrics['map50_95']:.4f} "
            f"best_{args.early_stop_metric}={best_metric:.4f} "
            f"no_improve={epochs_without_improvement}"
        )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            run_summary["stopped_early"] = True
            run_summary["stop_reason"] = (
                f"No validation {args.early_stop_metric} improvement for "
                f"{epochs_without_improvement} epochs"
            )
            print(f"[early-stop] {run_summary['stop_reason']}")
            break

    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, map_location=device)
    test_metrics = evaluate_detection(model, criterion, dataloaders["test"], device, args.img_size)
    run_summary["test"] = test_metrics
    run_summary["finished_at"] = time.time()
    run_summary["total_seconds"] = round(run_summary["finished_at"] - started_at, 2)
    save_json(metrics_file, run_summary)
    final_progress = progress_payload(
        args.method,
        run_summary["completed_epochs"],
        args.epochs,
        split_metrics={"test": test_metrics},
        started_at=started_at,
        stage="completed",
    )
    save_json(progress_file, final_progress)
    append_jsonl(history_file, final_progress)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
