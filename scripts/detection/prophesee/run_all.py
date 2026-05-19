import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHODS = [
    "ergo",
    "est",
    "evrepsl",
    "get",
    "event_pretraining",
    "matrix_lstm",
]


def run_command(command, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(str(part) for part in command), flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n# Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(" ".join(str(part) for part in command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def load_metrics(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def method_is_done(output_dir):
    metrics = load_metrics(Path(output_dir) / "metrics.json")
    return bool(metrics and metrics.get("test") is not None)


def build_index_if_needed(args):
    metadata_dir = Path(args.metadata_dir)
    needed = [metadata_dir / f"{split}.jsonl" for split in ("train", "val", "test")]
    if all(path.exists() for path in needed):
        print(f"[index] using existing metadata in {metadata_dir}", flush=True)
        return

    command = [
        sys.executable,
        "scripts/detection/prophesee/build_window_index.py",
        "--root",
        args.root,
        "--output-dir",
        args.metadata_dir,
        "--window-us",
        str(args.window_us),
        "--splits",
        "train",
        "val",
        "test",
    ]
    if args.index_max_files is not None:
        command.extend(["--max-files", str(args.index_max_files)])
    exit_code = run_command(command, Path(args.output_dir) / "build_index.log")
    if exit_code != 0:
        raise SystemExit(f"Index build failed with exit code {exit_code}")


def train_method(args, method):
    method_output_dir = Path(args.output_dir) / method
    if method_is_done(method_output_dir) and not args.force:
        print(f"[skip] {method}: metrics.json already has test results", flush=True)
        return

    command = [
        sys.executable,
        "scripts/detection/prophesee/train.py",
        "--root",
        args.root,
        "--method",
        method,
        "--metadata-dir",
        args.metadata_dir,
        "--config",
        args.config,
        "--output-dir",
        str(method_output_dir),
        "--window-us",
        str(args.window_us),
        "--img-size",
        str(args.img_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--sensor-width",
        str(args.sensor_width),
        "--sensor-height",
        str(args.sensor_height),
        "--detector-channels",
        str(args.detector_channels),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-metric",
        args.early_stop_metric,
        "--min-delta",
        str(args.min_delta),
    ]
    if args.resume:
        command.append("--resume")
    if args.cache_dir is not None:
        command.extend(["--cache-dir", args.cache_dir])
    if args.use_cache:
        command.append("--use-cache")
    if args.train_limit is not None:
        command.extend(["--train-limit", str(args.train_limit)])
    if args.val_limit is not None:
        command.extend(["--val-limit", str(args.val_limit)])
    if args.test_limit is not None:
        command.extend(["--test-limit", str(args.test_limit)])

    exit_code = run_command(command, method_output_dir / "train.log")
    if exit_code != 0:
        raise SystemExit(f"{method} failed with exit code {exit_code}")


def summarize(args):
    command = [
        sys.executable,
        "scripts/detection/prophesee/summarize.py",
        "--results-root",
        args.output_dir,
        "--output",
        str(Path(args.output_dir) / "summary.json"),
    ]
    exit_code = run_command(command, Path(args.output_dir) / "summary.log")
    if exit_code != 0:
        raise SystemExit(f"Summary failed with exit code {exit_code}")


def main():
    parser = argparse.ArgumentParser(description="Run Prophesee mini detection benchmark for selected methods.")
    parser.add_argument("--root", required=True, help="Dataset root containing train/val/test")
    parser.add_argument("--metadata-dir", default="metadata/prophesee_mini_windows")
    parser.add_argument("--output-dir", default="outputs/prophesee_mini")
    parser.add_argument("--config", default="configs/detection/yolov6n_prophesee.py")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--window-us", type=int, default=50_000)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-metric", default="map50_95", choices=["map50", "map50_95"])
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--detector-channels", type=int, default=12)
    parser.add_argument("--cache-dir", default=None, help="Optional precomputed tensor cache root")
    parser.add_argument("--use-cache", action="store_true", help="Train from cached tensors")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--index-max-files", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    build_index_if_needed(args)
    for method in args.methods:
        print(f"\n===== {method} =====", flush=True)
        train_method(args, method)
    summarize(args)
    print(f"[done] total_seconds={round(time.time() - started_at, 2)}", flush=True)


if __name__ == "__main__":
    main()
