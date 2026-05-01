import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


TRADITIONAL_METHODS = [
    "event_frame",
    "event_count",
    "binary_event_image",
    "timestamp_image",
    "time_surface",
    "voxel_grid",
]

DATASET_DEFAULTS = {
    "nmnist": {
        "tonic_name": "NMNIST",
        "num_classes": 10,
        "height": 34,
        "width": 34,
        "official_split": True,
    },
    "ncaltech101": {
        "tonic_name": "NCALTECH101",
        "num_classes": 101,
        "height": 180,
        "width": 240,
        "official_split": False,
        "split_file": "data/splits/tonic_split_seed42.json",
    },
}


def import_training_dependencies():
    """Import heavy training dependencies only when the script is actually run."""
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset

    try:
        import tonic
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'tonic'. Install it in the training environment, "
            "for example: python -m pip install tonic"
        ) from exc

    from src.models.classifier import EventClassifier
    from src.representations.registry import get_representation
    import src.representations.traditional  # noqa: F401 - registers methods

    return torch, DataLoader, Dataset, Subset, tonic, EventClassifier, get_representation


def set_seed(seed, torch=None):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_split_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [int(i) for i in payload["train"]], [int(i) for i in payload["test"]]


def split_train_val(indices, val_fraction, seed):
    indices = list(indices)
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(round(len(indices) * float(val_fraction)))
    val_size = max(1, val_size) if len(indices) > 1 and val_fraction > 0 else 0
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])
    return train_indices, val_indices


def sample_indices(length, limit):
    if limit is None:
        return list(range(length))
    return list(range(min(int(limit), length)))


def event_count(events):
    try:
        return int(len(events))
    except TypeError:
        return 0


class RepresentationStats:
    """Small running summary so the experiment is inspectable, not a black box."""

    def __init__(self):
        self.count = 0
        self.total_events = 0
        self.total_build_seconds = 0.0
        self.total_nonzero_ratio = 0.0
        self.total_min = 0.0
        self.total_max = 0.0
        self.total_mean = 0.0
        self.total_std = 0.0
        self.shape_counts = {}

    def update_batch(self, metas):
        for meta in metas:
            self.count += 1
            self.total_events += int(meta["num_events"])
            self.total_build_seconds += float(meta["build_seconds"])
            self.total_nonzero_ratio += float(meta["nonzero_ratio"])
            self.total_min += float(meta["min"])
            self.total_max += float(meta["max"])
            self.total_mean += float(meta["mean"])
            self.total_std += float(meta["std"])
            shape = "x".join(str(v) for v in meta["shape"])
            self.shape_counts[shape] = self.shape_counts.get(shape, 0) + 1

    def to_dict(self):
        denom = max(self.count, 1)
        return {
            "samples": self.count,
            "mean_events": round(self.total_events / denom, 4),
            "mean_build_seconds": round(self.total_build_seconds / denom, 6),
            "mean_nonzero_ratio": round(self.total_nonzero_ratio / denom, 6),
            "mean_min": round(self.total_min / denom, 6),
            "mean_max": round(self.total_max / denom, 6),
            "mean_value": round(self.total_mean / denom, 6),
            "mean_std": round(self.total_std / denom, 6),
            "shape_counts": dict(sorted(self.shape_counts.items())),
        }


def build_tonic_dataset(tonic, dataset_name, root, train):
    defaults = DATASET_DEFAULTS[dataset_name]
    dataset_cls = getattr(tonic.datasets, defaults["tonic_name"])

    # Tonic datasets differ a little. N-MNIST has an official train/test flag;
    # N-Caltech101 is one pool, so the split file handles train/test later.
    if defaults.get("official_split"):
        return dataset_cls(save_to=str(root), train=bool(train))
    return dataset_cls(save_to=str(root))


def encode_label(label, label_to_index=None):
    if isinstance(label, (int, np.integer)):
        return int(label)
    if label_to_index is not None and label in label_to_index:
        return int(label_to_index[label])
    try:
        return int(label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert dataset label {label!r} to an integer class id.") from exc


def normalize_tonic_sample(sample, label_to_index=None):
    if isinstance(sample, tuple) and len(sample) >= 2:
        return sample[0], encode_label(sample[1], label_to_index)
    if isinstance(sample, dict):
        label = sample.get("target", sample.get("label"))
        events = sample.get("events", sample.get("data"))
        return events, encode_label(label, label_to_index)
    raise TypeError(f"Unsupported tonic sample type: {type(sample)!r}")


def build_label_mapping(dataset):
    targets = getattr(dataset, "targets", None)
    if not targets:
        return None

    mapping = {}
    for label in sorted({target for target in targets if not isinstance(target, (int, np.integer))}, key=str):
        mapping[label] = len(mapping)
    return mapping or None


class TraditionalClassificationDataset:
    def __init__(self, base_dataset, representation, config, label_to_index=None):
        self.base_dataset = base_dataset
        self.representation = representation
        self.config = config
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        events, label = normalize_tonic_sample(self.base_dataset[idx], self.label_to_index)
        started = time.perf_counter()
        rep = self.representation.build(events)
        build_seconds = time.perf_counter() - started

        rep = np.asarray(rep, dtype=np.float32)
        if rep.ndim != 3:
            raise ValueError(f"Representation must be CxHxW, got shape {rep.shape}")

        meta = {
            "num_events": event_count(events),
            "build_seconds": build_seconds,
            "shape": list(rep.shape),
            "nonzero_ratio": float(np.count_nonzero(rep) / max(rep.size, 1)),
            "min": float(rep.min()) if rep.size else 0.0,
            "max": float(rep.max()) if rep.size else 0.0,
            "mean": float(rep.mean()) if rep.size else 0.0,
            "std": float(rep.std()) if rep.size else 0.0,
        }
        return rep, label, meta


def collate_samples(batch):
    import torch

    images = torch.from_numpy(np.stack([item[0] for item in batch], axis=0)).float()
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metas = [item[2] for item in batch]
    return images, labels, metas


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().sum().item(), labels.numel()


def run_one_epoch(torch, model, dataloader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0
    stats = RepresentationStats()
    started = time.perf_counter()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, metas in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            correct, seen = accuracy_from_logits(logits, labels)
            total_loss += float(loss.detach().item()) * seen
            total_correct += correct
            total_seen += seen
            stats.update_batch(metas)

    denom = max(total_seen, 1)
    return {
        "loss": round(total_loss / denom, 6),
        "accuracy": round(total_correct / denom, 6),
        "samples": total_seen,
        "seconds": round(time.perf_counter() - started, 2),
        "representation": stats.to_dict(),
    }


def evaluate_test(torch, model, dataloader, criterion, device):
    return run_one_epoch(torch, model, dataloader, criterion, device, optimizer=None)


def save_checkpoint(
    torch,
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    args_dict,
    best_epoch=None,
    epochs_without_improvement=0,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "args": args_dict,
        },
        path,
    )


def load_checkpoint(torch, path, model, optimizer=None, scheduler=None, device="cpu"):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def build_dataloaders(args, deps, representation):
    torch, DataLoader, Dataset, Subset, tonic, _, _ = deps
    dataset_name = args.dataset
    root = Path(args.root).expanduser().resolve()
    base_train = build_tonic_dataset(tonic, dataset_name, root, train=True)
    base_test = build_tonic_dataset(tonic, dataset_name, root, train=False)
    actual_split_file = None
    label_to_index = build_label_mapping(base_train)

    if DATASET_DEFAULTS[dataset_name].get("official_split"):
        train_indices = sample_indices(len(base_train), args.train_limit)
        test_indices = sample_indices(len(base_test), args.test_limit)
    else:
        split_file = Path(args.split_file or DATASET_DEFAULTS[dataset_name]["split_file"])
        actual_split_file = str(split_file)
        train_indices, test_indices = load_split_file(split_file)
        if args.train_limit is not None:
            train_indices = train_indices[: int(args.train_limit)]
        if args.test_limit is not None:
            test_indices = test_indices[: int(args.test_limit)]

    train_indices, val_indices = split_train_val(train_indices, args.val_fraction, args.seed)
    if args.val_limit is not None:
        val_indices = val_indices[: int(args.val_limit)]

    train_dataset = TraditionalClassificationDataset(Subset(base_train, train_indices), representation, vars(args), label_to_index)
    val_dataset = TraditionalClassificationDataset(Subset(base_train, val_indices), representation, vars(args), label_to_index)
    test_dataset = TraditionalClassificationDataset(Subset(base_test, test_indices), representation, vars(args), label_to_index)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_samples,
    }
    return {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    }, {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "split_file": actual_split_file,
        "val_fraction": args.val_fraction,
        "label_to_index": label_to_index,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Traditional event-representation classification baseline.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS), help="Dataset to train on.")
    parser.add_argument("--root", required=True, help="Tonic dataset cache/root directory.")
    parser.add_argument("--method", required=True, choices=TRADITIONAL_METHODS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-file", default=None, help="Split JSON for datasets without official train/test.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--tau-us", type=float, default=None)
    parser.add_argument("--max-events", type=int, default=50000)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    return parser.parse_args()


def log_line(path, message):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


def main():
    args = parse_args()
    deps = import_training_dependencies()
    torch, _, _, _, _, EventClassifier, get_representation = deps
    set_seed(args.seed, torch)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available. Use --device cpu for local smoke tests.")
    device = torch.device(args.device)
    defaults = DATASET_DEFAULTS[args.dataset]
    height = args.height or defaults["height"]
    width = args.width or defaults["width"]
    output_dir = Path(args.output_dir or Path("outputs") / "traditional" / "classification" / args.dataset / args.method)
    output_dir.mkdir(parents=True, exist_ok=True)

    representation_config = {
        "height": height,
        "width": width,
        "bins": args.bins,
        "num_bins": args.bins,
        "tau_us": args.tau_us,
        "max_events": args.max_events,
        "normalize": args.normalize,
    }
    representation_cls = get_representation(args.method)
    representation = representation_cls(representation_config)

    dataloaders, data_summary = build_dataloaders(args, deps, representation)
    first_batch = next(iter(dataloaders["train"]))
    input_channels = int(first_batch[0].shape[1])
    model = EventClassifier(in_channels=input_channels, num_classes=defaults["num_classes"]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)

    config_payload = {
        "dataset": args.dataset,
        "method": args.method,
        "command": " ".join(sys.argv),
        "root": str(Path(args.root).expanduser().resolve()),
        "dataset_defaults": defaults,
        "data": data_summary,
        "representation": representation_config | {"input_channels": input_channels},
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "device": str(device),
            "seed": args.seed,
            "early_stop_patience": args.early_stop_patience,
            "min_delta": args.min_delta,
            "downstream_model": "ResNet18",
        },
    }
    save_json(output_dir / "config.json", config_payload)

    checkpoints_dir = output_dir / "checkpoints"
    last_ckpt = checkpoints_dir / "last.pt"
    best_ckpt = checkpoints_dir / "best.pt"
    history_path = output_dir / "history.jsonl"
    progress_path = output_dir / "progress.json"
    metrics_path = output_dir / "metrics.json"
    representation_stats_path = output_dir / "representation_stats.json"
    log_path = output_dir / "train.log"

    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0
    started_at = time.time()

    if args.resume and last_ckpt.exists():
        payload = load_checkpoint(torch, last_ckpt, model, optimizer, scheduler, device=device)
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload.get("best_metric", best_metric))
        best_epoch = payload.get("best_epoch")
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))

    run_summary = config_payload | {
        "train": [],
        "val": [],
        "test": None,
        "best_epoch": best_epoch,
        "best_val_accuracy": None,
        "started_at": started_at,
    }

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_one_epoch(torch, model, dataloaders["train"], criterion, device, optimizer=optimizer)
        val_metrics = run_one_epoch(torch, model, dataloaders["val"], criterion, device, optimizer=None)
        scheduler.step()

        current_metric = float(val_metrics["accuracy"])
        improved = current_metric > best_metric + args.min_delta
        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(
                torch,
                best_ckpt,
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                vars(args),
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            torch,
            last_ckpt,
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            vars(args),
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
        )
        epoch_payload = {
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "best_epoch": best_epoch,
            "best_val_accuracy": None if best_metric == float("-inf") else round(best_metric, 6),
            "epochs_without_improvement": epochs_without_improvement,
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        run_summary["train"].append(train_metrics)
        run_summary["val"].append(val_metrics)
        run_summary["best_epoch"] = best_epoch
        run_summary["best_val_accuracy"] = None if best_metric == float("-inf") else round(best_metric, 6)
        append_jsonl(history_path, epoch_payload)
        save_json(progress_path, epoch_payload | {"stage": "training"})
        save_json(metrics_path, run_summary | {"updated_at": time.time()})

        log_line(
            log_path,
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"best_val_acc={best_metric:.4f} no_improve={epochs_without_improvement}",
        )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            run_summary["stopped_early"] = True
            run_summary["stop_reason"] = f"No val accuracy improvement for {epochs_without_improvement} epochs"
            log_line(log_path, f"[early-stop] {run_summary['stop_reason']}")
            break

    if best_ckpt.exists():
        load_checkpoint(torch, best_ckpt, model, device=device)
    test_metrics = evaluate_test(torch, model, dataloaders["test"], criterion, device)
    run_summary["test"] = test_metrics
    run_summary["finished_at"] = time.time()
    run_summary["total_seconds"] = round(run_summary["finished_at"] - started_at, 2)
    representation_stats = {
        "train_last_epoch": run_summary["train"][-1]["representation"] if run_summary["train"] else None,
        "val_last_epoch": run_summary["val"][-1]["representation"] if run_summary["val"] else None,
        "test": test_metrics["representation"],
    }
    save_json(representation_stats_path, representation_stats)
    save_json(metrics_path, run_summary)
    save_json(progress_path, {"stage": "completed", "test": test_metrics, "best_epoch": best_epoch})
    log_line(log_path, "[completed] " + json.dumps({"test": test_metrics, "best_epoch": best_epoch}, ensure_ascii=True))
    print(json.dumps(run_summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
