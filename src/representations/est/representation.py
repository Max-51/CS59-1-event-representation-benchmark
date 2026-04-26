"""
EST-style Dense Event Tensor Representation
============================================

IMPORTANT NOTE:
    This is an EST-style dense event tensor adaptation for the benchmark pipeline,
    not a byte-to-byte reproduction of the original learnable EST module.

    The original EST (ICCV 2019, "End-to-End Learning of Representations for
    Asynchronous Event-Based Data", Gehrig et al.) uses a learnable QuantizationLayer
    (MLP-based) to compute per-event soft bin weights in an end-to-end trainable manner.

    This implementation uses deterministic bilinear temporal interpolation instead,
    which produces a structurally similar (2*num_bins, H, W) voxel grid without
    requiring the learnable MLP. This design choice ensures a self-contained,
    dependency-free benchmark baseline that is fair to compare against other methods.

    implementation_type: EST-style dense event tensor adaptation
    Reference: https://github.com/uzh-rpg/rpg_event_representation_learning

Output shape: (2 * num_bins, height, width)
    channels   0 ..   num_bins-1 : positive polarity temporal bins
    channels num_bins .. 2*num_bins-1 : negative polarity temporal bins
"""

import numpy as np
import torch

from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


@register_representation("est")
class ESTRepresentation(BaseRepresentation):
    """
    EST-style bilinear event voxel grid representation.

    Args:
        config (dict): Configuration dictionary with keys:
            num_bins   (int)  : Number of temporal bins. Default: 9.
            height     (int)  : Sensor/output height in pixels. Default: 180.
            width      (int)  : Sensor/output width  in pixels. Default: 240.
            max_events (int)  : Maximum number of events to use per sample.
                                Events are randomly sub-sampled if exceeded.
                                Default: 50000.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_bins   = config.get("num_bins",   9)
        self.height     = config.get("height",    180)
        self.width      = config.get("width",     240)
        self.max_events = config.get("max_events", 50000)

    # ------------------------------------------------------------------
    def _parse_events(self, events):
        """Parse events from Tonic structured array or plain (N,4) ndarray."""
        if hasattr(events, "dtype") and events.dtype.names is not None:
            # Tonic returns structured numpy array with named fields x/y/t/p
            x = events["x"].astype(np.int64)
            y = events["y"].astype(np.int64)
            t = events["t"].astype(np.float64)
            p = events["p"].astype(np.int64)   # 0 (neg) or 1 (pos)
        else:
            # Fallback: plain (N, 4) array with columns [x, y, t, p]
            events = np.asarray(events)
            x = events[:, 0].astype(np.int64)
            y = events[:, 1].astype(np.int64)
            t = events[:, 2].astype(np.float64)
            p = events[:, 3].astype(np.int64)
        return x, y, t, p

    # ------------------------------------------------------------------
    def build(self, events):
        """
        Build the EST-style voxel grid from a stream of events.

        Args:
            events: Tonic structured numpy array (fields: x, y, t, p)
                    or plain numpy ndarray of shape (N, 4).

        Returns:
            torch.FloatTensor of shape (2 * num_bins, height, width).
            All values are in [-1, 1] after per-channel safe normalisation.
        """
        C = 2 * self.num_bins

        # --- Guard: empty input ---
        if events is None or len(events) == 0:
            return torch.zeros(C, self.height, self.width, dtype=torch.float32)

        x, y, t, p = self._parse_events(events)

        # --- Guard: still empty after parse ---
        if len(x) == 0:
            return torch.zeros(C, self.height, self.width, dtype=torch.float32)

        # ① Random sub-sampling if event count exceeds max_events
        #    Sample without replacement, then sort to preserve temporal order.
        if len(x) > self.max_events:
            idx = np.random.choice(len(x), self.max_events, replace=False)
            idx.sort()
            x, y, t, p = x[idx], y[idx], t[idx], p[idx]

        # ② Filter out-of-bounds coordinates
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]

        # --- Guard: all events out of bounds ---
        if len(x) == 0:
            return torch.zeros(C, self.height, self.width, dtype=torch.float32)

        # ③ Normalise timestamps to [0, num_bins - 1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min) * (self.num_bins - 1)
        else:
            # All events at the same time → place in bin 0
            t_norm = np.zeros(len(t), dtype=np.float64)

        # ④ Bilinear temporal interpolation
        #    Each event contributes to two adjacent bins with complementary weights.
        #    t_norm ∈ [0, num_bins-1]
        #    bin_floor ∈ [0, num_bins-1],  bin_ceil = bin_floor + 1 ∈ [1, num_bins]
        voxel = np.zeros((C, self.height, self.width), dtype=np.float32)

        bin_floor = np.floor(t_norm).astype(np.int64)             # lower bin index
        bin_floor = np.clip(bin_floor, 0, self.num_bins - 1)
        bin_ceil  = bin_floor + 1                                  # upper bin index

        w_ceil  = (t_norm - bin_floor).astype(np.float32)         # weight for upper bin
        w_floor = (1.0 - w_ceil).astype(np.float32)               # weight for lower bin

        # Polarity channel offset:
        #   p == 1 (positive) → channels   0 ..   num_bins-1   (offset = 0)
        #   p == 0 (negative) → channels num_bins .. 2*num_bins-1 (offset = num_bins)
        pol_offset = (p == 0).astype(np.int64) * self.num_bins

        # ---- Floor-bin contribution (always valid: bin_floor < num_bins) ----
        ch_f = pol_offset + bin_floor
        np.add.at(voxel, (ch_f, y, x), w_floor)

        # ---- Ceil-bin contribution (valid only when bin_ceil < num_bins) ----
        ceil_valid = bin_ceil < self.num_bins
        if ceil_valid.any():
            ch_c = (pol_offset + bin_ceil)[ceil_valid]
            np.add.at(voxel,
                      (ch_c, y[ceil_valid], x[ceil_valid]),
                      w_ceil[ceil_valid])

        # ⑤ Safe global normalisation: divide by absolute maximum
        voxel_max = np.abs(voxel).max()
        if voxel_max > 0.0:
            voxel /= voxel_max

        return torch.from_numpy(voxel)  # (2*num_bins, H, W)
