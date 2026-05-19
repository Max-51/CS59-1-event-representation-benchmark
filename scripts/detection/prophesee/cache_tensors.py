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
from src.detection.prophesee.benchmark import PropheseeYoloV6Dataset


DEFAULT_METHODS = [
    "ergo",
    "est",
    "evrepsl",
    "get",
    "event_pretraining",
    "matrix_lstm",
]


def ensure_indices(root, metadata_dir, window_us, max_files=None):
    root = Path(root).resolve()
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_paths = {}
    for split in ("train", "val", "test"):
        index_path = metadata_dir / f"{split}.jsonl"
        if not index_path.exists():
            entries = build_prophesee_window_index(root, split, window_us=window_us, max_files=max_files)
            write_prophesee_window_index(entries, index_path, root=root)
        index_paths[split] = index_path.resolve()
    return index_paths


def tensor_to_dtype(tensor, dtype):
    if dtype == "float16":
        return tensor.half()
    if dtype == "float32":
        return tensor.float()
    raise ValueError(f"Unsupported cache dtype: {dtype}")


def cache_split(args, method, split, index_path, limit):
    split_dir = Path(args.cache_dir) / method / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.force:
        print(f"[skip] cache exists: {manifest_path}", flush=True)
        return

    dataset = PropheseeYoloV6Dataset(
        index_path,
        method,
        root=args.root,
        representation_config={"device": args.representation_device, "return_numpy": False},
        img_size=args.img_size,
        detector_channels=args.detector_channels,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        max_windows=limit,
    )

    started = time.perf_counter()
    bytes_written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx in range(len(dataset)):
            item = dataset[idx]
            filename = f"{idx:08d}.pt"
            payload = {
                "image": tensor_to_dtype(item["image"], args.cache_dtype),
                "labels": item["labels"].float(),
                "recording_id": item["recording_id"],
                "window": tuple(item["window"]),
                "original_shape": tuple(item["original_shape"]),
                "resized_shape": tuple(item["resized_shape"]),
            }
            output_path = split_dir / filename
            torch.save(payload, output_path)
            bytes_written += output_path.stat().st_size
            manifest.write(json.dumps({"index": idx, "file": filename}, ensure_ascii=True) + "\n")
            if (idx + 1) % args.log_every == 0 or idx + 1 == len(dataset):
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = (idx + 1) / elapsed
                print(
                    f"[{method}/{split}] {idx + 1}/{len(dataset)} "
                    f"rate={rate:.2f} samples/s cache={bytes_written / 1024**3:.2f} GiB",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description="Precompute Prophesee mini YOLOv6 tensors for faster repeated training.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--metadata-dir", default="metadata/prophesee_mini_windows")
    parser.add_argument("--cache-dir", default="cache/prophesee_mini_tensors")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--window-us", type=int, default=50_000)
    parser.add_argument("--index-max-files", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--detector-channels", type=int, default=12)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--representation-device", default="cpu")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    args.root = str(Path(args.root).resolve())
    class_names = read_dataset_label_map(args.root)
    index_paths = ensure_indices(args.root, args.metadata_dir, args.window_us, max_files=args.index_max_files)
    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}

    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "cache_config.json").write_text(
        json.dumps(
            {
                "root": args.root,
                "class_names": list(class_names),
                "methods": args.methods,
                "splits": args.splits,
                "window_us": args.window_us,
                "img_size": args.img_size,
                "sensor_width": args.sensor_width,
                "sensor_height": args.sensor_height,
                "detector_channels": args.detector_channels,
                "cache_dtype": args.cache_dtype,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for method in args.methods:
        for split in args.splits:
            cache_split(args, method, split, index_paths[split], limits[split])


if __name__ == "__main__":
    main()
