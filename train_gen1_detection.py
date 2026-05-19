#!/usr/bin/env python3
"""Compatibility launcher for the reorganized GEN1 detection entrypoint."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "tasks"
    / "detection"
    / "scripts"
    / "train_gen1_detection.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
