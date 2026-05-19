#!/usr/bin/env python3
"""
EST End-to-End classification training script.
Learnable QuantizationLayer (MLP) is trained jointly with ResNet backbone.

Examples:
    # N-Caltech101
    python train_est_e2e_classification.py --dataset ncaltech101 \
        --data_root /content/ncaltech101 --checkpoint_dir /content/ckpts/est_e2e

    # N-MNIST
    python train_est_e2e_classification.py --dataset nmnist \
        --data_root /content/nmnist --checkpoint_dir /content/ckpts/est_e2e_nmnist
"""

import argparse, json, os, random, datetime, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import tonic
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import src.datasets.cifar10dvs_patch  # local .aedat loader


DATASET_DEFAULTS = {
    "ncaltech101": {
        "tonic_cls"     : "NCALTECH101",
        "num_classes"   : 101,
        "height"        : 180,
        "width"         : 240,
        "official_split": False,
    },
    "nmnist": {
        "tonic_cls"     : "NMNIST",
        "num_classes"   : 10,
        "height"        : 34,
        "width"         : 34,
        "official_split": True,
    },
    "cifar10dvs": {
        "tonic_cls"     : "CIFAR10DVS",
        "num_classes"   : 10,
        "height"        : 128,
        "width"         : 128,
        "official_split": False,
    },
}

DATASET_ALIASES = {
    "minist": "nmnist",
    "ncar101": "ncaltech101",
    "cifa": "cifar10dvs",
    "cifar": "cifar10dvs",
    "cifar10-dvs": "cifar10dvs",
    "cifar10_dvs": "cifar10dvs",
    "n-caltech101": "ncaltech101",
    "n-caltech": "ncaltech101",
    "n-mnist": "nmnist",
}


def normalize_dataset_name(name):
    key = str(name).strip().lower()
    return DATASET_ALIASES.get(key, key)


# ── Dataset ────────────────────────────────────────────────────────────────────

class ESTRawEventDataset(Dataset):
    """Returns raw events as (max_events, 5) tensor [x, y, t, p, valid_mask]."""

    def __init__(self, tonic_dataset, max_events, height, width, class_to_idx=None):
        self.dataset      = tonic_dataset
        self.max_events   = max_events
        self.height       = height
        self.width        = width
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        events, label = self.dataset[idx]

        if hasattr(events, "dtype") and events.dtype.names is not None:
            x = events["x"].astype(np.float32)
            y = events["y"].astype(np.float32)
            t = events["t"].astype(np.float32)
            p = events["p"].astype(np.float32)
        else:
            arr = np.asarray(events, dtype=np.float32)
            x, y, t, p = arr[:,0], arr[:,1], arr[:,2], arr[:,3]

        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]
        N = len(x)

        M = self.max_events
        if N == 0:
            ev_out    = np.zeros((M, 4), dtype=np.float32)
            vmask_out = np.zeros(M,      dtype=np.float32)
        elif N >= M:
            sel = np.random.choice(N, M, replace=False)
            sel.sort()
            ev_out    = np.stack([x[sel], y[sel], t[sel], p[sel]], axis=1)
            vmask_out = np.ones(M, dtype=np.float32)
        else:
            pad_idx  = np.random.choice(N, M - N, replace=True)
            base     = np.stack([x, y, t, p], axis=1)
            ev_out   = np.concatenate([base, base[pad_idx]], axis=0)
            vmask_out = np.zeros(M, dtype=np.float32)
            vmask_out[:N] = 1.0

        ev_with_mask = np.concatenate(
            [ev_out, vmask_out.reshape(-1, 1)], axis=1
        )  # (M, 5)

        if isinstance(label, str) and self.class_to_idx is not None:
            label = self.class_to_idx[label]

        return torch.from_numpy(ev_with_mask), int(label)


def _build_class_to_idx(tonic_dataset):
    for attr in ("label_to_int", "class_to_idx"):
        if hasattr(tonic_dataset, attr):
            val = getattr(tonic_dataset, attr)
            if isinstance(val, dict) and val:
                return dict(val)
    labels = set()
    for i in range(len(tonic_dataset)):
        _, lbl = tonic_dataset[i]
        labels.add(lbl)
    return {cls: idx for idx, cls in enumerate(sorted(labels))}


def build_dataloaders(args, ds_cfg):
    tonic_cls = getattr(tonic.datasets, ds_cfg["tonic_cls"])

    if ds_cfg["official_split"]:
        train_tonic = tonic_cls(save_to=args.data_root, train=True)
        test_tonic  = tonic_cls(save_to=args.data_root, train=False)
        train_set = ESTRawEventDataset(
            train_tonic, args.max_events, ds_cfg["height"], ds_cfg["width"]
        )
        test_set = ESTRawEventDataset(
            test_tonic, args.max_events, ds_cfg["height"], ds_cfg["width"]
        )
    else:
        all_tonic    = tonic_cls(save_to=args.data_root)
        class_to_idx = _build_class_to_idx(all_tonic)
        with open(args.split_file) as f:
            splits = json.load(f)
        full_ds   = ESTRawEventDataset(
            all_tonic, args.max_events, ds_cfg["height"], ds_cfg["width"],
            class_to_idx=class_to_idx
        )
        train_set = Subset(full_ds, splits["train"])
        test_set  = Subset(full_ds, splits["test"])

    def seed_worker(worker_id):
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, test_loader


# ── Train / Eval ───────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ev_batch, labels in loader:
            ev_batch, labels = ev_batch.to(device), labels.to(device)
            correct += (model(ev_batch).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def run_training(args, model, train_loader, test_loader, device):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path   = os.path.join(args.checkpoint_dir, "best_model.pth")
    latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    start_epoch    = 0
    best_acc       = 0.0
    no_improve_cnt = 0
    train_history  = []
    stopped_early  = False

    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch    = ckpt["epoch"] + 1
        best_acc       = ckpt.get("best_acc", 0.0)
        no_improve_cnt = ckpt.get("no_improve_cnt", 0)
        train_history  = ckpt.get("train_history", [])
        print(f"Resumed from epoch {ckpt['epoch']}  best_acc={best_acc:.4f}")

    print(f"Training EST-E2E on {args.dataset} | "
          f"epochs={args.epochs} bs={args.batch_size} "
          f"lr={args.lr} patience={args.patience} device={device}")
    print("-" * 70)

    last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        model.train()
        run_loss = correct = total = 0

        for ev_batch, labels in train_loader:
            ev_batch, labels = ev_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(ev_batch)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        scheduler.step()
        train_loss = run_loss / total
        train_acc  = correct / total
        test_acc   = evaluate(model, test_loader, device)
        lr_now     = optimizer.param_groups[0]["lr"]

        if test_acc > best_acc:
            best_acc       = test_acc
            no_improve_cnt = 0
            flag           = "★"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_acc": best_acc}, best_path)
        else:
            no_improve_cnt += 1
            flag = f"(no improve {no_improve_cnt}/{args.patience})"

        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"loss={train_loss:.4f} train={train_acc:.4f} "
              f"test={test_acc:.4f} lr={lr_now:.2e} {flag}")

        train_history.append({
            "epoch": epoch + 1, "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4),
            "no_improve_cnt": no_improve_cnt,
        })
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc, "no_improve_cnt": no_improve_cnt,
            "train_history": train_history,
        }, latest_path)

        if no_improve_cnt >= args.patience:
            print(f"\n⚑ Early stopping at epoch {epoch+1}.")
            stopped_early = True
            break

    return best_acc, train_history, last_epoch + 1, stopped_early


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="ncaltech101")
    parser.add_argument("--data_root",      required=True)
    parser.add_argument("--split_file",     default=str(PROJECT_ROOT / "data" / "splits" / "tonic_split_seed42.json"))
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir",    default="artifacts/classification/learning")
    parser.add_argument("--backbone",       default="resnet18",
                        choices=["resnet18", "resnet34"])
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_bins",     type=int,   default=9)
    parser.add_argument("--max_events",   type=int,   default=50000)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    if args.dataset not in DATASET_DEFAULTS:
        valid = ", ".join(sorted(DATASET_DEFAULTS))
        raise SystemExit(f"Unknown dataset {args.dataset!r}. Valid datasets: {valid}")

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = DATASET_DEFAULTS[args.dataset]
    train_loader, test_loader = build_dataloaders(args, ds_cfg)

    from src.models.est_e2e import ESTEndToEndClassifier
    model = ESTEndToEndClassifier(
        num_bins    = args.num_bins,
        height      = ds_cfg["height"],
        width       = ds_cfg["width"],
        num_classes = ds_cfg["num_classes"],
        backbone    = args.backbone,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ESTEndToEndClassifier | "
          f"num_bins={args.num_bins} | params={total_params/1e6:.4f}M")

    best_acc, train_history, actual_epochs, stopped_early = run_training(
        args, model, train_loader, test_loader, device
    )

    # save results
    results_dir = os.path.join(args.results_dir, "est")
    os.makedirs(results_dir, exist_ok=True)
    result_name = f"est_e2e_{args.dataset}"

    results = {
        "method"             : "EST (End-to-End, learnable QuantizationLayer)",
        "implementation_type": "Original EST reproduction — QuantizationLayer jointly trained with ResNet",
        "dataset"            : args.dataset,
        "epochs_planned"     : args.epochs,
        "epochs_actual"      : actual_epochs,
        "early_stopping"     : {"patience": args.patience, "triggered": stopped_early},
        "batch_size"         : args.batch_size,
        "lr"                 : args.lr,
        "weight_decay"       : args.weight_decay,
        "seed"               : args.seed,
        "num_bins"           : args.num_bins,
        "max_events"         : args.max_events,
        "backbone"           : args.backbone,
        "num_classes"        : ds_cfg["num_classes"],
        "best_test_accuracy"     : round(best_acc, 4),
        "best_test_accuracy_pct" : round(best_acc * 100, 2),
        "gpu" : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "train_history"      : train_history,
        "timestamp"          : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = os.path.join(results_dir, f"{result_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Best test accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"✓ Results saved: {json_path}")


if __name__ == "__main__":
    main()
