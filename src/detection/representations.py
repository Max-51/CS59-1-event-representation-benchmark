import numpy as np

from src.representations.traditional import (
    BinaryEventImageRepresentation,
    EventFrameRepresentation,
    TimeSurfaceRepresentation,
    TimestampImageRepresentation,
    VoxelGridRepresentation,
)


def normalized_events(events, width, height, polarity="zero_one"):
    arr = np.asarray(events)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if arr.dtype.names is not None:
        x = arr["x"].astype(np.float32)
        y = arr["y"].astype(np.float32)
        t = arr["t"].astype(np.float32)
        p = arr["p"].astype(np.float32)
    else:
        x = arr[:, 0].astype(np.float32)
        y = arr[:, 1].astype(np.float32)
        t = arr[:, 2].astype(np.float32)
        p = arr[:, 3].astype(np.float32)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    if polarity == "minus_one_one":
        p = np.where(p > 0, 1.0, -1.0).astype(np.float32)
    else:
        p = (p > 0).astype(np.float32)
    order = np.argsort(t, kind="stable")
    return np.stack([x[order], y[order], t[order], p[order]], axis=1).astype(np.float32)


def normalize_channels(chw):
    out = chw.astype(np.float32, copy=True)
    for channel in range(out.shape[0]):
        values = out[channel]
        mask = values != 0
        if np.any(mask):
            scale = np.max(np.abs(values[mask]))
            if scale > 0:
                out[channel] = values / scale
    return out


class ESTDetectionRepresentation:
    def __init__(self, config):
        self.height = int(config.get("height", 240))
        self.width = int(config.get("width", 304))
        self.bins = int(config.get("bins", config.get("num_bins", 9)))

    def build(self, events):
        events = normalized_events(events, self.width, self.height, polarity="zero_one")
        output = np.zeros((2 * self.bins, self.height, self.width), dtype=np.float32)
        if len(events) == 0:
            return output
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = events[:, 2]
        p = events[:, 3].astype(np.int64)
        span = max(float(t[-1] - t[0]), 1.0)
        tbin = (t - t[0]) / span * max(self.bins - 1, 1)
        lo = np.floor(tbin).astype(np.int64)
        hi = np.clip(lo + 1, 0, self.bins - 1)
        w_hi = (tbin - lo).astype(np.float32)
        w_lo = 1.0 - w_hi
        offset = (1 - p) * self.bins
        np.add.at(output, (offset + lo, y, x), w_lo)
        np.add.at(output, (offset + hi, y, x), w_hi)
        return normalize_channels(output)


class ERGODetectionRepresentation:
    def __init__(self, config):
        self.height = int(config.get("height", 240))
        self.width = int(config.get("width", 304))
        self.stack_size = int(config.get("stack_size", 12))

    def build(self, events):
        events = normalized_events(events, self.width, self.height, polarity="minus_one_one")
        output = np.zeros((self.stack_size, self.height, self.width), dtype=np.float32)
        if len(events) == 0:
            return output
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = events[:, 2]
        p = events[:, 3]
        span = max(float(t[-1] - t[0]), 1.0)
        bins = np.clip(((t - t[0]) / span * self.stack_size).astype(np.int64), 0, self.stack_size - 1)
        np.add.at(output, (bins, y, x), p)
        return normalize_channels(output)


class EvRepSLDetectionRepresentation:
    def __init__(self, config):
        self.height = int(config.get("height", 240))
        self.width = int(config.get("width", 304))

    def build(self, events):
        events = normalized_events(events, self.width, self.height, polarity="minus_one_one")
        output = np.zeros((3, self.height, self.width), dtype=np.float32)
        if len(events) == 0:
            return output
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = events[:, 2]
        p = events[:, 3]
        span = max(float(t[-1] - t[0]), 1.0)
        tn = (t - t[0]) / span
        np.add.at(output[0], (y, x), 1.0)
        np.add.at(output[1], (y, x), p)
        np.maximum.at(output[2], (y, x), tn.astype(np.float32))
        return normalize_channels(output)


class TokenGridDetectionRepresentation:
    def __init__(self, config):
        self.height = int(config.get("height", 240))
        self.width = int(config.get("width", 304))
        self.group_num = int(config.get("group_num", 12))
        self.patch_size = int(config.get("patch_size", 4))

    def build(self, events):
        events = normalized_events(events, self.width, self.height, polarity="zero_one")
        patch = max(self.patch_size, 1)
        out_h = int(np.ceil(self.height / patch))
        out_w = int(np.ceil(self.width / patch))
        time_bins = max(1, self.group_num // 2)
        channels = time_bins * 4
        output = np.zeros((out_h, out_w, channels), dtype=np.float32)
        if len(events) == 0:
            return output
        x = np.clip((events[:, 0] // patch).astype(np.int64), 0, out_w - 1)
        y = np.clip((events[:, 1] // patch).astype(np.int64), 0, out_h - 1)
        t = events[:, 2]
        p = events[:, 3].astype(np.int64)
        span = max(float(t[-1] - t[0]), 1.0)
        tn = (t - t[0]) / span
        tb = np.clip((tn * time_bins).astype(np.int64), 0, time_bins - 1)
        count_ch = tb * 4 + p
        time_ch = tb * 4 + 2 + p
        np.add.at(output, (y, x, count_ch), 1.0)
        np.add.at(output, (y, x, time_ch), tn.astype(np.float32))
        return output.astype(np.float32)


class MatrixLSTMDetectionRepresentation:
    def __init__(self, config):
        self.height = int(config.get("height", 240))
        self.width = int(config.get("width", 304))
        self.hidden_size = int(config.get("hidden_size", 12))

    def build(self, events):
        events = normalized_events(events, self.width, self.height, polarity="minus_one_one")
        output = np.zeros((self.hidden_size, self.height, self.width), dtype=np.float32)
        if len(events) == 0:
            return output
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = events[:, 2]
        p = events[:, 3]
        span = max(float(t[-1] - t[0]), 1.0)
        tn = (t - t[0]) / span
        features = [
            np.ones_like(tn),
            p,
            tn,
            1.0 - tn,
            p * tn,
            p * (1.0 - tn),
        ]
        for channel in range(self.hidden_size):
            values = features[channel % len(features)]
            np.add.at(output[channel], (y, x), values.astype(np.float32))
        return normalize_channels(output)


class TraditionalDetectionRepresentation:
    def __init__(self, representation_cls, config):
        self.representation = representation_cls(config)

    def build(self, events):
        tensor = self.representation.build(events)
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy().astype(np.float32)
        return np.asarray(tensor, dtype=np.float32)


def create_detection_representation(method, config):
    method = method.lower()
    if method == "est":
        return ESTDetectionRepresentation(config)
    if method == "ergo":
        return ERGODetectionRepresentation(config)
    if method == "evrepsl":
        return EvRepSLDetectionRepresentation(config)
    if method == "get":
        return TokenGridDetectionRepresentation(config)
    if method == "event_pretraining":
        return TokenGridDetectionRepresentation(config)
    if method == "matrix_lstm":
        return MatrixLSTMDetectionRepresentation(config)
    if method in ("event_frame", "event_count"):
        return TraditionalDetectionRepresentation(EventFrameRepresentation, config)
    if method == "binary_event_image":
        return TraditionalDetectionRepresentation(BinaryEventImageRepresentation, config)
    if method == "timestamp_image":
        return TraditionalDetectionRepresentation(TimestampImageRepresentation, config)
    if method == "time_surface":
        return TraditionalDetectionRepresentation(TimeSurfaceRepresentation, config)
    if method == "voxel_grid":
        return TraditionalDetectionRepresentation(VoxelGridRepresentation, config)
    raise KeyError(f"Unknown detection representation method: {method}")

