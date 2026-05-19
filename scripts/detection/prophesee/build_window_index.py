import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.prophesee_detection import build_prophesee_window_index, write_prophesee_window_index


def main():
    parser = argparse.ArgumentParser(description="Build Prophesee detection window metadata.")
    parser.add_argument("--root", required=True, help="Dataset root containing train/val/test folders")
    parser.add_argument("--output-dir", default="metadata/prophesee_mini_windows")
    parser.add_argument("--window-us", type=int, default=50_000)
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap per split for debugging")
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
        entries = build_prophesee_window_index(root, split, window_us=args.window_us, max_files=args.max_files)
        output_path = write_prophesee_window_index(entries, output_dir / f"{split}.jsonl", root=root)
        summary["splits"][split] = {
            "windows": len(entries),
            "seconds": round(time.perf_counter() - start, 2),
            "output": str(output_path.resolve()),
        }
        print(f"[{split}] windows={len(entries)} output={output_path}", flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
