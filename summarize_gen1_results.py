#!/usr/bin/env python3
"""Compatibility launcher for the reorganized GEN1 summary entrypoint."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "tasks"
    / "detection"
    / "scripts"
    / "summarize_gen1_results.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
