import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml, random, argparse
import numpy as np
from pathlib import Path
from src.datasets.ncaltech101 import NCaltech101
from src.models.classifier import EventClassifier
from src.representations.registry import get_representation
from src.representations.est.representation import ESTRepresentation
from src.representations.ergo.representation import ERGORepresentation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  [Checkpoint saved → {path}]")

def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_acc"]

def train(config_path, data_root=None, resume=None, save_every=5):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if data_root:
        cfg["dataset"]["root"] = data_root

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    RepClass = get_representation(cfg["method"])
    rep = RepClass(cfg["representation"])
    def transform(events): return rep.build(events)

    train_set = NCaltech101(cfg["dataset"]["root"], cfg["dataset"]["train_split"], transform=transform)
    test_set  = NCaltech101(cfg["dataset"]["root"], cfg["dataset"]["test_split"],  transform=transform)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # 兼容 EST (num_bins * 2) 和 ERGO (in_channels)
    rep_cfg = cfg["representation"]
    in_channels = rep_cfg.get("in_channels", rep_cfg.get("num_bins", 5) * 2)

    model = EventClassifier(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    results_dir = Path("results") / cfg["method"]
    results_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_acc = 0.0

    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            start_epoch, best_acc = load_checkpoint(resume_path, model, optimizer, scheduler)
            print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")
        else:
            print(f"WARNING: checkpoint {resume} not found, starting from scratch.")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for events, labels in train_loader:
            events, labels = events.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(events)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_acc = correct / total * 100

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for events, labels in test_loader:
                events, labels = events.to(device), labels.to(device)
                outputs = model(events)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total * 100

        print(f"Epoch {epoch+1}/{cfg['epochs']} | Loss: {total_loss/len(train_loader):.3f} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), results_dir / "best_model.pth")

        if (epoch + 1) % save_every == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }, results_dir / "checkpoint_latest.pth")

    print(f"\n最终最佳准确率: {best_acc:.2f}%")
    with open(results_dir / "result.txt", "w") as f:
        f.write(f"method: {cfg['method']}\nbest_acc: {best_acc:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()
    train(args.config, args.data_root, args.resume, args.save_every)