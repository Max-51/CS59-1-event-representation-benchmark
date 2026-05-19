#!/usr/bin/env python3
"""Compatibility launcher for the reorganized GEN1 indexing script."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent.parent
    / "tasks"
    / "detection"
    / "scripts"
    / "build_gen1_window_index.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
