import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.prophesee_detection import (
    build_prophesee_window_index,
    read_dataset_label_map,
    write_prophesee_window_index,
)
from src.detection.prophesee.benchmark import (
    CachedYoloV6Dataset,
    PropheseeYoloV6Dataset,
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
            entries = build_prophesee_window_index(root, split, window_us=window_us)
            write_prophesee_window_index(entries, index_path, root=root)
        index_paths[split] = index_path.resolve()
    return index_paths


def _metric_value(metrics, metric_name):
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(f"Metric '{metric_name}' was not produced by evaluation")
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv6 on Prophesee mini detection windows.")
    parser.add_argument("--root", required=True, help="Dataset root containing train/val/test")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--metadata-dir", default="metadata/prophesee_mini_windows")
    parser.add_argument("--config", default="configs/detection/yolov6n_prophesee.py")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window-us", type=int, default=50_000)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--detector-channels", type=int, default=12)
    parser.add_argument("--cache-dir", default=None, help="Optional precomputed tensor cache root")
    parser.add_argument("--use-cache", action="store_true", help="Train from cached tensors instead of raw DAT windows")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-metric", default="map50_95", choices=["map50", "map50_95"])
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = Path(args.root).resolve()
    metadata_dir = Path(args.metadata_dir)
    config_path = Path(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "prophesee_mini" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_dataset_label_map(root)
    index_paths = ensure_indices(root, metadata_dir, args.window_us)
    if args.use_cache:
        if args.cache_dir is None:
            raise ValueError("--use-cache requires --cache-dir")
        datasets = {
            "train": CachedYoloV6Dataset(args.cache_dir, args.method, "train", max_windows=args.train_limit),
            "val": CachedYoloV6Dataset(args.cache_dir, args.method, "val", max_windows=args.val_limit),
            "test": CachedYoloV6Dataset(args.cache_dir, args.method, "test", max_windows=args.test_limit),
        }
    else:
        representation_config = {"device": str(device), "return_numpy": False}
        dataset_kwargs = {
            "root": root,
            "representation_config": representation_config,
            "img_size": args.img_size,
            "detector_channels": args.detector_channels,
            "sensor_width": args.sensor_width,
            "sensor_height": args.sensor_height,
        }
        datasets = {
            "train": PropheseeYoloV6Dataset(
                index_paths["train"],
                args.method,
                max_windows=args.train_limit,
                **dataset_kwargs,
            ),
            "val": PropheseeYoloV6Dataset(
                index_paths["val"],
                args.method,
                max_windows=args.val_limit,
                **dataset_kwargs,
            ),
            "test": PropheseeYoloV6Dataset(
                index_paths["test"],
                args.method,
                max_windows=args.test_limit,
                **dataset_kwargs,
            ),
        }
    dataloaders = {
        "train": create_dataloader(datasets["train"], args.batch_size, True, args.num_workers),
        "val": create_dataloader(datasets["val"], args.batch_size, False, args.num_workers),
        "test": create_dataloader(datasets["test"], args.batch_size, False, args.num_workers),
    }

    _, model, criterion = build_yolov6_model(
        config_path,
        device,
        args.img_size,
        num_classes=len(class_names),
        number_of_channels=args.detector_channels,
    )
    optimizer = create_optimizer(model, args.lr, args.momentum, args.weight_decay)
    scheduler = create_scheduler(optimizer, args.epochs)
    started_at = time.time()
    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0

    progress_file = output_dir / "progress.json"
    metrics_file = output_dir / "metrics.json"
    history_file = output_dir / "history.jsonl"
    checkpoints_dir = output_dir / "checkpoints"
    last_ckpt = checkpoints_dir / "last.pt"
    best_ckpt = checkpoints_dir / "best.pt"

    if args.resume and last_ckpt.exists():
        resume_payload = load_checkpoint(last_ckpt, model, optimizer, scheduler, map_location=device)
        start_epoch = int(resume_payload["epoch"]) + 1
        best_metric = float(resume_payload.get("best_metric", best_metric))
        best_epoch = resume_payload.get("best_epoch")
        epochs_without_improvement = int(resume_payload.get("epochs_without_improvement", 0))

    run_summary = {
        "dataset": "prophesee_mini_detection",
        "method": args.method,
        "root": str(root),
        "metadata_dir": str(metadata_dir.resolve()),
        "config": str(config_path.resolve()),
        "class_names": list(class_names),
        "device": str(device),
        "sensor_width": args.sensor_width,
        "sensor_height": args.sensor_height,
        "img_size": args.img_size,
        "detector_channels": args.detector_channels,
        "cache_dir": args.cache_dir,
        "use_cache": args.use_cache,
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

    save_json(progress_file, progress_payload(args.method, start_epoch, args.epochs, started_at=started_at, stage="initializing"))

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
        val_metrics = evaluate_detection(
            model,
            criterion,
            dataloaders["val"],
            device,
            args.img_size,
            class_names=class_names,
        )
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

        ckpt = checkpoint_payload(epoch, model, optimizer, scheduler, best_metric, vars(args).copy())
        ckpt["best_metric_name"] = args.early_stop_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            run_summary["best_epoch"] = best_epoch
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

        run_summary["completed_epochs"] = epoch + 1
        run_summary["best_epoch"] = best_epoch
        run_summary["best_metric"] = None if best_metric == float("-inf") else best_metric
        ckpt["best_metric"] = best_metric
        ckpt["best_epoch"] = best_epoch
        ckpt["best_map50"] = run_summary["best_map50"]
        ckpt["best_map50_95"] = run_summary["best_map50_95"]
        ckpt["epochs_without_improvement"] = epochs_without_improvement
        save_checkpoint(last_ckpt, ckpt)
        save_json(metrics_file, run_summary | {"updated_at": time.time()})

        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_metrics['mean_total_loss']:.4f} "
            f"val_map50={val_metrics['map50']:.4f} "
            f"val_map50_95={val_metrics['map50_95']:.4f} "
            f"best_{args.early_stop_metric}={best_metric:.4f} "
            f"no_improve={epochs_without_improvement}",
            flush=True,
        )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            run_summary["stopped_early"] = True
            run_summary["stop_reason"] = (
                f"No validation {args.early_stop_metric} improvement for "
                f"{epochs_without_improvement} epochs"
            )
            print(f"[early-stop] {run_summary['stop_reason']}", flush=True)
            break

    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, map_location=device)
    test_metrics = evaluate_detection(
        model,
        criterion,
        dataloaders["test"],
        device,
        args.img_size,
        class_names=class_names,
    )
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
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
