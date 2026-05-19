#!/usr/bin/env python3
"""Compatibility bridge for the reorganized traditional classification entrypoint."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "tasks"
    / "classification"
    / "scripts"
    / "train_traditional_classification.py"
)

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
else:
    globals().update(runpy.run_path(str(TARGET)))
