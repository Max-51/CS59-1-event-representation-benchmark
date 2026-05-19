#!/usr/bin/env python3
"""Compatibility launcher for the reorganized GEN1 full benchmark entrypoint."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "tasks"
    / "detection"
    / "scripts"
    / "run_all_gen1_methods.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
