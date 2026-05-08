import numpy as np
import torch

from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


@register_representation("omnievent")
class OmniEventRepresentation(BaseRepresentation):
    def __init__(self, config):
        super().__init__(config)
        self.height = int(config.get("height", 180))
        self.width = int(config.get("width", 240))
        self.max_events = int(config.get("max_events", 50000))
        self.channels = int(config.get("channels", 8))

    def build(self, events):
        c = self.channels
        if events is None or len(events) == 0:
            return torch.zeros((c, self.height, self.width), dtype=torch.float32)

        if hasattr(events, "dtype") and events.dtype.names is not None:
            x = events["x"].astype(np.int64)
            y = events["y"].astype(np.int64)
            t = events["t"].astype(np.float64)
            p = events["p"].astype(np.int64)
        else:
            arr = np.asarray(events)
            x = arr[:, 0].astype(np.int64)
            y = arr[:, 1].astype(np.int64)
            t = arr[:, 2].astype(np.float64)
            p = arr[:, 3].astype(np.int64)

        if len(x) > self.max_events:
            idx = np.random.choice(len(x), self.max_events, replace=False)
            idx.sort()
            x, y, t, p = x[idx], y[idx], t[idx], p[idx]

        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]
        if len(x) == 0:
            return torch.zeros((c, self.height, self.width), dtype=torch.float32)

        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min)
        else:
            t_norm = np.zeros_like(t)

        rep = np.zeros((c, self.height, self.width), dtype=np.float32)
        pos = p > 0
        neg = ~pos
        sign = np.where(pos, 1.0, -1.0).astype(np.float32)

        np.add.at(rep[0], (y, x), 1.0)
        np.add.at(rep[1], (y[pos], x[pos]), 1.0)
        np.add.at(rep[2], (y[neg], x[neg]), 1.0)
        np.add.at(rep[3], (y, x), sign)
        np.add.at(rep[4], (y, x), t_norm.astype(np.float32))
        np.add.at(rep[5], (y, x), (t_norm ** 2).astype(np.float32))

        bins = max(c - 6, 1)
        g = np.clip(np.floor(t_norm * (bins - 1)).astype(np.int64), 0, bins - 1)
        for i in range(bins):
            ch = 6 + i
            if ch >= c:
                break
            mask = g == i
            if mask.any():
                np.add.at(rep[ch], (y[mask], x[mask]), 1.0)

        max_abs = np.abs(rep).max()
        if max_abs > 0:
            rep /= max_abs
        return torch.from_numpy(rep)
