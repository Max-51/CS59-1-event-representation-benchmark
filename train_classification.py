#!/usr/bin/env python3
"""
Unified classification training script.
Supports: est, ergo, evrepsl, event_pretraining, matrix_lstm

Examples:
    # N-Caltech101
    python train_classification.py --method est --dataset ncaltech101 \
        --data_root /content/ncaltech101 --checkpoint_dir /content/ckpts/est

    # N-MNIST
    python train_classification.py --method est --dataset nmnist \
        --data_root /content/nmnist --checkpoint_dir /content/ckpts/est_nmnist

    # EvRepSL (requires RepGen weights)
    python train_classification.py --method evrepsl --dataset ncaltech101 \
        --data_root /content/ncaltech101 --checkpoint_dir /content/ckpts/evrepsl \
        --repgen_path /content/weights/RepGen.pth \
        --evrepsl_repo /content/evrepsl_repo
"""

import argparse, json, os, random, datetime, sys
import numpy as np
import torch
import torch.nn as nn
import tonic
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR


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
}


class EventDataset(Dataset):
    def __init__(self, tonic_dataset, representation, class_to_idx=None):
        self.dataset      = tonic_dataset
        self.rep          = representation
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        events, label = self.dataset[idx]
        tensor = self.rep.build(events)
        if isinstance(label, str) and self.class_to_idx is not None:
            label = self.class_to_idx[label]
        return tensor, int(label)


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


def build_dataloaders(args, representation, ds_cfg):
    tonic_cls = getattr(tonic.datasets, ds_cfg["tonic_cls"])

    if ds_cfg["official_split"]:
        train_tonic = tonic_cls(save_to=args.data_root, train=True)
        test_tonic  = tonic_cls(save_to=args.data_root, train=False)
        train_set   = EventDataset(train_tonic, representation)
        test_set    = EventDataset(test_tonic,  representation)
    else:
        all_tonic    = tonic_cls(save_to=args.data_root)
        class_to_idx = _build_class_to_idx(all_tonic)
        with open(args.split_file) as f:
            splits = json.load(f)
        full_ds   = EventDataset(all_tonic, representation, class_to_idx)
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


def evaluate(model, loader, device, repgen=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if repgen is not None:
                imgs = repgen(imgs)
            preds    = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def run_training(args, model, train_loader, test_loader, device, repgen=None):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path   = os.path.join(args.checkpoint_dir, "best_model.pth")
    latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    start_epoch   = 0
    best_acc      = 0.0
    train_history = []

    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_acc      = ckpt.get("best_acc", 0.0)
        train_history = ckpt.get("train_history", [])
        print(f"Resumed from epoch {ckpt['epoch']} (best_acc={best_acc:.4f})")

    print(f"Training {args.method} on {args.dataset} | "
          f"epochs={args.epochs} lr={args.lr} bs={args.batch_size} device={device}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if repgen is not None:
                with torch.no_grad():
                    imgs = repgen(imgs)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc  = correct / total
        test_acc   = evaluate(model, test_loader, device, repgen)
        lr_now     = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"loss={train_loss:.4f} train={train_acc:.4f} "
              f"test={test_acc:.4f} lr={lr_now:.2e}")

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc" : round(train_acc,  4),
            "test_acc"  : round(test_acc,   4),
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_acc": best_acc}, best_path)
            print(f"  ★ New best test_acc={best_acc:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc"            : best_acc,
            "train_history"       : train_history,
        }, latest_path)

    return best_acc, train_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",         required=True,
                        choices=["est","ergo","evrepsl","event_pretraining","matrix_lstm"])
    parser.add_argument("--dataset",        default="ncaltech101",
                        choices=["ncaltech101","nmnist"])
    parser.add_argument("--data_root",      required=True)
    parser.add_argument("--split_file",     default="data/splits/tonic_split_seed42.json")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir",    default="results/classification")
    parser.add_argument("--repgen_path",    default=None)
    parser.add_argument("--evrepsl_repo",   default=None)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--max_events",   type=int,   default=50000)
    parser.add_argument("--num_bins",     type=int,   default=9,
                        help="EST only: number of temporal bins")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = DATASET_DEFAULTS[args.dataset]

    # representation
    from src.representations.registry import get_representation
    rep_cfg = {"height": ds_cfg["height"], "width": ds_cfg["width"],
               "max_events": args.max_events}
    if args.method == "est":
        rep_cfg["num_bins"] = args.num_bins
    representation = get_representation(args.method)(rep_cfg)

    # dataloaders
    train_loader, test_loader = build_dataloaders(args, representation, ds_cfg)

    # determine in_channels from first sample
    base_ds   = train_loader.dataset
    first_ds  = base_ds.dataset if isinstance(base_ds, Subset) else base_ds
    first_idx = base_ds.indices[0] if isinstance(base_ds, Subset) else 0
    sample_tensor, _ = first_ds[first_idx]
    in_channels = sample_tensor.shape[0]

    # RepGen for EvRepSL
    repgen = None
    if args.method == "evrepsl" and args.repgen_path:
        if args.evrepsl_repo:
            sys.path.insert(0, args.evrepsl_repo)
        from models import EffWNet
        repgen = EffWNet(n_channels=3, out_depth=1, inc_f0=1, bilinear=True,
                         n_lyr=4, ch1=12, c_is_const=False, c_is_scalar=False,
                         device=str(device))
        repgen.load_state_dict(torch.load(args.repgen_path, map_location=device))
        repgen.to(device).eval()
        for p in repgen.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels,
                                ds_cfg["height"], ds_cfg["width"]).to(device)
            in_channels = repgen(dummy).shape[1]
        print(f"EvRepSL: RepGen loaded, in_channels after RepGen = {in_channels}")

    # model
    from src.models.classifier import EventClassifier
    model = EventClassifier(in_channels=in_channels,
                            num_classes=ds_cfg["num_classes"]).to(device)
    print(f"Model: EventClassifier(in_channels={in_channels}, "
          f"num_classes={ds_cfg['num_classes']})")

    # train
    best_acc, train_history = run_training(
        args, model, train_loader, test_loader, device, repgen
    )

    # save results
    results_dir = os.path.join(args.results_dir, args.method)
    os.makedirs(results_dir, exist_ok=True)
    result_name = f"{args.method}_{args.dataset}"
    results = {
        "method": args.method, "dataset": args.dataset,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "seed": args.seed, "in_channels": in_channels,
        "num_classes": ds_cfg["num_classes"],
        "best_test_accuracy"    : round(best_acc, 4),
        "best_test_accuracy_pct": round(best_acc * 100, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "train_history": train_history,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = os.path.join(results_dir, f"{result_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Best test accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"✓ Results saved: {json_path}")


if __name__ == "__main__":
    main()
