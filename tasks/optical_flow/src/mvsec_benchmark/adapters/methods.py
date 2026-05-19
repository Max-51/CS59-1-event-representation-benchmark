from __future__ import annotations

from typing import Any

from .base import AdapterSpec
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


def build_adapters() -> dict[str, Any]:
    """Return the local runnable adapter registry."""
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
