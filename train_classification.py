#!/usr/bin/env python3
"""Compatibility launcher for the reorganized classification entrypoint."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "tasks"
    / "classification"
    / "scripts"
    / "train_classification.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
