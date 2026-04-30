"""Traditional event representations for the unified benchmark interface.

All classes accept event streams as either a structured array with fields
``x, y, t, p`` or a plain ``Nx4`` array in ``[x, y, t, p]`` order, and return a
``numpy.float32`` array in ``C x H x W`` layout.
"""

import numpy as np

from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


def _parse_events(events, width, height, max_events=None):
    if events is None or len(events) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    arr = np.asarray(events)
    if arr.dtype.names is not None:
        x = arr["x"].astype(np.int64)
        y = arr["y"].astype(np.int64)
        t = arr["t"].astype(np.float64)
        p = arr["p"].astype(np.int64)
    else:
        x = arr[:, 0].astype(np.int64)
        y = arr[:, 1].astype(np.int64)
        t = arr[:, 2].astype(np.float64)
        p = arr[:, 3].astype(np.int64)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    if len(x) == 0:
        return x, y, t, p

    order = np.argsort(t, kind="stable")
    x, y, t, p = x[order], y[order], t[order], p[order]
    if max_events is not None and len(x) > int(max_events):
        x = x[-int(max_events) :]
        y = y[-int(max_events) :]
        t = t[-int(max_events) :]
        p = p[-int(max_events) :]

    p = np.where(p > 0, 1, 0).astype(np.int64)
    return x, y, t, p


def _polarity_channels(p):
    """Map polarity to channel offset: positive first, negative second."""
    return np.where(p > 0, 0, 1).astype(np.int64)


def _normalize_channels(chw):
    out = chw.astype(np.float32, copy=True)
    for channel in range(out.shape[0]):
        values = out[channel]
        scale = np.max(np.abs(values))
        if scale > 0:
            out[channel] = values / scale
    return out


class TraditionalRepresentation(BaseRepresentation):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(config)
        self.height = int(config.get("height", 180))
        self.width = int(config.get("width", 240))
        self.max_events = config.get("max_events", 50000)
        self.normalize = bool(config.get("normalize", True))

    def _events(self, events):
        return _parse_events(events, self.width, self.height, self.max_events)

    def _finish(self, output):
        if self.normalize:
            output = _normalize_channels(output)
        return output.astype(np.float32, copy=False)


@register_representation("event_count")
@register_representation("event_frame")
class EventFrameRepresentation(TraditionalRepresentation):
    """Two-channel event count image: channel 0 positive, channel 1 negative."""

    def build(self, events):
        x, y, _, p = self._events(events)
        output = np.zeros((2, self.height, self.width), dtype=np.float32)
        if len(x) == 0:
            return self._finish(output)

        channels = _polarity_channels(p)
        np.add.at(output, (channels, y, x), 1.0)
        return self._finish(output)


@register_representation("binary_event_image")
class BinaryEventImageRepresentation(TraditionalRepresentation):
    """Two-channel binary occupancy image: channel 0 positive, channel 1 negative."""

    def build(self, events):
        x, y, _, p = self._events(events)
        output = np.zeros((2, self.height, self.width), dtype=np.float32)
        if len(x) == 0:
            return self._finish(output)

        channels = _polarity_channels(p)
        output[channels, y, x] = 1.0
        return self._finish(output)


@register_representation("timestamp_image")
class TimestampImageRepresentation(TraditionalRepresentation):
    """Latest timestamp image: channel 0 positive, channel 1 negative."""

    def build(self, events):
        x, y, t, p = self._events(events)
        output = np.zeros((2, self.height, self.width), dtype=np.float32)
        if len(x) == 0:
            return self._finish(output)

        span = max(float(t[-1] - t[0]), 1.0)
        t_norm = ((t - t[0]) / span).astype(np.float32)
        channels = _polarity_channels(p)
        np.maximum.at(output, (channels, y, x), t_norm)
        return output.astype(np.float32, copy=False)


@register_representation("time_surface")
class TimeSurfaceRepresentation(TraditionalRepresentation):
    """Exponential time surface split by polarity.

    Each active pixel stores ``exp(-(t_end - t_last) / tau)``. If ``tau_us`` is
    not provided, the representation uses 30% of the current window duration.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.tau_us = None if self.config.get("tau_us") is None else float(self.config["tau_us"])

    def build(self, events):
        x, y, t, p = self._events(events)
        latest = np.full((2, self.height, self.width), -np.inf, dtype=np.float64)
        if len(x) == 0:
            return self._finish(np.zeros_like(latest, dtype=np.float32))

        channels = _polarity_channels(p)
        np.maximum.at(latest, (channels, y, x), t)
        span = max(float(t[-1] - t[0]), 1.0)
        tau = self.tau_us if self.tau_us and self.tau_us > 0 else 0.3 * span
        output = np.zeros_like(latest, dtype=np.float32)
        active = np.isfinite(latest)
        output[active] = np.exp(-(float(t[-1]) - latest[active]) / max(tau, 1.0)).astype(np.float32)
        return output.astype(np.float32, copy=False)


@register_representation("voxel_grid")
class VoxelGridRepresentation(TraditionalRepresentation):
    """Voxel grid with positive bins first, then negative bins."""

    def __init__(self, config=None):
        super().__init__(config)
        self.num_bins = int(self.config.get("num_bins", self.config.get("bins", 5)))

    def build(self, events):
        x, y, t, p = self._events(events)
        output = np.zeros((2 * self.num_bins, self.height, self.width), dtype=np.float32)
        if len(x) == 0:
            return self._finish(output)

        span = max(float(t[-1] - t[0]), 1.0)
        tbin = (t - t[0]) / span * max(self.num_bins - 1, 1)
        lo = np.clip(np.floor(tbin).astype(np.int64), 0, self.num_bins - 1)
        hi = np.clip(lo + 1, 0, self.num_bins - 1)
        w_hi = (tbin - lo).astype(np.float32)
        w_lo = 1.0 - w_hi
        offset = _polarity_channels(p) * self.num_bins

        np.add.at(output, (offset + lo, y, x), w_lo)
        np.add.at(output, (offset + hi, y, x), w_hi)
        return self._finish(output)
