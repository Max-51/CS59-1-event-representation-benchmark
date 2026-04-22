import numpy as np
import torch
from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation

@register_representation("ergo")
class ERGORepresentation(BaseRepresentation):
    """
    ERGO-12: From Chaos Comes Order (ICCV 2023)
    Ordered event representation with 12 channels via Gryffin-optimized configuration.
    """

    WINDOW_INDEXES = [0, 3, 2, 2, 0, 0, 1, 0, 1, 0, 0, 1]
    FUNCTIONS = [
        "polarity", "timestamp_neg", "count_neg", "count_neg", "count_pos",
        "timestamp_pos", "timestamp_neg", "count_pos", "count_pos",
        "timestamp", "timestamp", "timestamp_pos"
    ]
    AGGREGATIONS = [
        "variance", "variance", "mean", "mean", "mean",
        "mean", "sum", "max", "sum", "max", "variance", "variance"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.height = config.get("height", 180)
        self.width  = config.get("width",  240)

    def _scatter(self, values, indices, aggregation):
        size = self.height * self.width
        if aggregation == "sum":
            out = np.zeros(size, dtype=np.float64)
            np.add.at(out, indices, values)
        elif aggregation == "mean":
            s = np.zeros(size, dtype=np.float64)
            c = np.zeros(size, dtype=np.float64)
            np.add.at(s, indices, values)
            np.add.at(c, indices, 1.0)
            out = np.where(c > 0, s / c, 0.0)
        elif aggregation == "max":
            out = np.full(size, -np.inf, dtype=np.float64)
            np.maximum.at(out, indices, values)
            out[out == -np.inf] = 0.0
        elif aggregation == "variance":
            s  = np.zeros(size, dtype=np.float64)
            s2 = np.zeros(size, dtype=np.float64)
            c  = np.zeros(size, dtype=np.float64)
            np.add.at(s,  indices, values)
            np.add.at(s2, indices, values ** 2)
            np.add.at(c,  indices, 1.0)
            mean    = np.where(c > 0, s  / c, 0.0)
            mean_sq = np.where(c > 0, s2 / c, 0.0)
            out = mean_sq - mean ** 2
        return out.reshape(self.height, self.width)

    def _surface(self, x, y, t, p, func, aggregation):
        if len(x) == 0:
            return np.zeros((self.height, self.width), dtype=np.float64)
        idx = x + y * self.width
        if func == "timestamp":
            return self._scatter(t, idx, aggregation)
        elif func == "polarity":
            return self._scatter(p, idx, aggregation)
        elif func == "count":
            return self._scatter(np.ones(len(x)), idx, aggregation)
        elif func == "timestamp_pos":
            m = p == 1
            return self._scatter(t[m], idx[m], aggregation) if m.any() else np.zeros((self.height, self.width))
        elif func == "timestamp_neg":
            m = p == 0
            return self._scatter(t[m], idx[m], aggregation) if m.any() else np.zeros((self.height, self.width))
        elif func == "count_pos":
            m = p == 1
            return self._scatter(np.ones(m.sum()), idx[m], aggregation) if m.any() else np.zeros((self.height, self.width))
        elif func == "count_neg":
            m = p == 0
            return self._scatter(np.ones(m.sum()), idx[m], aggregation) if m.any() else np.zeros((self.height, self.width))
        return np.zeros((self.height, self.width))

    def _create_windows(self, x, y, t, p):
        """7 SBN windows: 1 full + 3 equal chunks by count + 3 progressive halvings."""
        n = len(x)
        chunk = max(n // 3, 1)
        windows = [(x, y, t, p)]
        for i in range(3):
            sl = slice(i * chunk, (i + 1) * chunk)
            windows.append((x[sl], y[sl], t[sl], p[sl]))
        xh, yh, th, ph = x.copy(), y.copy(), t.copy(), p.copy()
        for _ in range(3):
            cut = max(len(xh) // 2, 1)
            xh, yh, th, ph = xh[cut:], yh[cut:], th[cut:], ph[cut:]
            windows.append((xh, yh, th, ph))
        return windows

    def build(self, events):
        """
        Args:
            events: np.ndarray (N, 4) — [x, y, t, p], p in {0, 1}
        Returns:
            torch.FloatTensor (12, H, W)
        """
        if len(events) == 0:
            return torch.zeros(12, self.height, self.width)
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2].astype(np.float64)
        p = events[:, 3].astype(np.float64)
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        windows = self._create_windows(x, y, t, p)
        channels = [
            self._surface(*windows[wi], func, agg)
            for wi, func, agg in zip(self.WINDOW_INDEXES, self.FUNCTIONS, self.AGGREGATIONS)
        ]
        return torch.FloatTensor(np.stack(channels, axis=0))