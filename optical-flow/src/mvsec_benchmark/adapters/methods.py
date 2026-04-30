from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import AdapterSpec
from .common import voxel_count_representation
from .event_pretraining import EventPretrainingAdapter
from .ergo import ErgoAdapter
from .evrepsl import EvRepSLAdapter
from .est import EstAdapter
from .get import GetAdapter
from .matrixlstm import MatrixLSTMAdapter
from .traditional import (
    BinaryEventImageAdapter,
    EventFrameAdapter,
    TimeSurfaceAdapter,
    TimestampImageAdapter,
    VoxelGridAdapter,
)


@dataclass
class SimpleAdapter:
    spec: AdapterSpec
    time_bins: int = 4
    split_polarity: bool = True

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        return voxel_count_representation(
            events,
            sensor_size=sensor_size,
            time_bins=self.time_bins,
            split_polarity=self.split_polarity,
        )


def build_adapters() -> dict[str, Any]:
    """Return all seven method adapters.

    The benchmark currently mixes first-pass paper-aware adapters with a few
    remaining placeholders. That is intentional: every method gets a stable
    integration point early, then we progressively replace scaffold logic with
    paper-specific representation builders.
    """
    return {
        "est": EstAdapter(
            AdapterSpec("est", channels=18, source_status="native-flow-paper", notes="NumPy EST-style trilinear quantization"),
            time_bins=9,
        ),
        "ergo": ErgoAdapter(
            AdapterSpec("ergo", channels=12, source_status="adapted-flow", notes="NumPy ERGO optimized representation adapter"),
        ),
        "event_pretraining": EventPretrainingAdapter(
            AdapterSpec("event_pretraining", channels=2, source_status="native-flow-paper", notes="Two-channel pos/neg event frame from upstream pretraining loader"),
        ),
        "get": GetAdapter(
            AdapterSpec("get", channels=24, source_status="adapted-flow", notes="GET-style grouped event tokens unwrapped to dense flow features"),
        ),
        "matrixlstm": MatrixLSTMAdapter(
            AdapterSpec("matrixlstm", channels=4, source_status="native-flow-paper", notes="Per-pixel sequence summary inspired by MatrixLSTM optical-flow defaults"),
        ),
        "evrepsl": EvRepSLAdapter(
            AdapterSpec("evrepsl", channels=3, source_status="native-flow-paper", notes="EvRep fallback with optional RepGen upgrade"),
        ),
        "omnievent": SimpleAdapter(
            AdapterSpec("omnievent", channels=12, source_status="native-flow-paper-reported", notes="OmniEvent placeholder until upstream code matures"),
            time_bins=6,
            split_polarity=True,
        ),
        "event_frame": EventFrameAdapter(
            AdapterSpec("event_frame", channels=2, source_status="traditional-baseline", notes="Two-channel positive/negative event count frame"),
        ),
        "event_count": EventFrameAdapter(
            AdapterSpec("event_count", channels=2, source_status="traditional-baseline", notes="Alias for two-channel event count frame"),
        ),
        "binary_event_image": BinaryEventImageAdapter(
            AdapterSpec("binary_event_image", channels=2, source_status="traditional-baseline", notes="Two-channel positive/negative binary occupancy image"),
        ),
        "timestamp_image": TimestampImageAdapter(
            AdapterSpec("timestamp_image", channels=2, source_status="traditional-baseline", notes="Latest normalized timestamp image split by polarity"),
        ),
        "time_surface": TimeSurfaceAdapter(
            AdapterSpec("time_surface", channels=2, source_status="traditional-baseline", notes="Exponential recency surface split by polarity"),
        ),
        "voxel_grid": VoxelGridAdapter(
            AdapterSpec("voxel_grid", channels=10, source_status="traditional-baseline", notes="Five-bin polarity-separated voxel grid"),
            bins=5,
        ),
    }
