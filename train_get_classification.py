#!/usr/bin/env python3
"""
GET (Group Event Transformer) classification training script.
Requires the official GET repo: https://github.com/Peterande/GET-Group-Event-Transformer

Setup:
    git clone https://github.com/Peterande/GET-Group-Event-Transformer.git /content/GET-repo
    python train_get_classification.py \
        --data_root /content/ncaltech101 \
        --get_repo  /content/GET-repo \
        --checkpoint_dir /content/ckpts/get

Note:
    GET uses its own transformer backbone (GroupEventTransformer), not ResNet18.
    This script interfaces directly with the official GET implementation.
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
        "tonic_cls": "NCALTECH101", "num_classes": 101,
        "height": 180, "width": 240, "official_split": False,
    },
    "nmnist": {
        "tonic_cls": "NMNIST", "num_classes": 10,
        "height": 34, "width": 34, "official_split": True,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="ncaltech101",
                        choices=["ncaltech101", "nmnist"])
    parser.add_argument("--data_root",      required=True)
    parser.add_argument("--get_repo",       required=True,
                        help="Path to cloned GET official repo")
    parser.add_argument("--split_file",     default="data/splits/tonic_split_seed42.json")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir",    default="results/classification")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    sys.path.insert(0, args.get_repo)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = DATASET_DEFAULTS[args.dataset]

    # ── dataset & model: adapt from GET official code ──────────────────────
    # TODO: wire up GET's own dataset pipeline and GroupEventTransformer model
    # Reference: {args.get_repo}/main.py or train.py in the GET repo
    raise NotImplementedError(
        "Please adapt this script using GET's official training pipeline. "
        f"See: {args.get_repo}"
    )


if __name__ == "__main__":
    main()
