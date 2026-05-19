from torch.utils.data import Dataset
import torch
import json
from pathlib import Path

from src.datasets.prophesee_detection import PropheseeIndexedWindowDataset
from src.detection.yolov6_training import (
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
from src.detection.prophesee.yolov6 import PropheseeYoloV6SampleBuilder


class PropheseeYoloV6Dataset(Dataset):
    def __init__(
        self,
        index_path,
        method,
        root=None,
        representation_config=None,
        img_size=320,
        detector_channels=12,
        sensor_width=1280,
        sensor_height=720,
        max_windows=None,
    ):
        self.base = PropheseeIndexedWindowDataset(index_path, root=root)
        self.limit = len(self.base) if max_windows is None else min(len(self.base), int(max_windows))
        self.builder = PropheseeYoloV6SampleBuilder(
            method,
            representation_config=representation_config,
            img_size=img_size,
            detector_channels=detector_channels,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
        )

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        return self.builder.build(self.base[idx])


class CachedYoloV6Dataset(Dataset):
    """Dataset backed by precomputed YOLOv6-ready tensors."""

    def __init__(self, cache_dir, method, split, max_windows=None):
        self.cache_dir = Path(cache_dir) / method / split
        self.manifest_path = self.cache_dir / "manifest.jsonl"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Cache manifest not found: {self.manifest_path}. "
                "Run scripts/detection/prophesee/cache_tensors.py first."
            )
        self.items = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        if max_windows is not None:
            self.items = self.items[: int(max_windows)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        payload = torch.load(self.cache_dir / item["file"], map_location="cpu")
        image = payload["image"].float()
        labels = payload["labels"].float()
        return {
            "image": image,
            "labels": labels,
            "recording_id": payload["recording_id"],
            "window": tuple(payload["window"]),
            "original_shape": tuple(payload["original_shape"]),
            "resized_shape": tuple(payload["resized_shape"]),
        }


__all__ = [
    "PropheseeYoloV6Dataset",
    "CachedYoloV6Dataset",
    "append_jsonl",
    "build_yolov6_model",
    "checkpoint_payload",
    "create_dataloader",
    "create_optimizer",
    "create_scheduler",
    "evaluate_detection",
    "load_checkpoint",
    "progress_payload",
    "save_checkpoint",
    "save_json",
    "train_one_epoch",
]
