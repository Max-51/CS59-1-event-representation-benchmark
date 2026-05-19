#!/usr/bin/env python3
"""Compatibility launcher for the reorganized EST E2E entrypoint."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "tasks"
    / "classification"
    / "scripts"
    / "train_est_e2e_classification.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
