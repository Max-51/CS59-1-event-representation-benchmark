import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.gen1_detection import build_gen1_window_index, write_gen1_window_index


def main():
    parser = argparse.ArgumentParser(description="Build unified GEN1 window metadata index.")
    parser.add_argument("--root", required=True, help="GEN1 extracted dataset root")
    parser.add_argument("--output-dir", default="metadata/gen1_windows", help="Directory for train/val/test jsonl indices")
    parser.add_argument("--window-us", type=int, default=50_000, help="Fixed temporal window in microseconds")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap per split for quick debugging")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "root": str(root),
        "window_us": args.window_us,
        "max_files": args.max_files,
        "splits": {},
    }

    for split in args.splits:
        start = time.perf_counter()
        entries = build_gen1_window_index(root, split, window_us=args.window_us, max_files=args.max_files)
        output_path = write_gen1_window_index(entries, output_dir / f"{split}.jsonl", root=root)
        summary["splits"][split] = {
            "windows": len(entries),
            "seconds": round(time.perf_counter() - start, 2),
            "output": str(output_path.resolve()),
        }
        print(f"[{split}] windows={len(entries)} output={output_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
