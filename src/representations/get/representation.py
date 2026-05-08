import numpy as np
import torch

from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


@register_representation("get")
class GETRepresentation(BaseRepresentation):
    def __init__(self, config):
        super().__init__(config)
        self.height = int(config.get("height", 180))
        self.width = int(config.get("width", 240))
        self.max_events = int(config.get("max_events", 50000))
        self.group_num = int(config.get("group_num", 12))

    def build(self, events):
        c = self.group_num
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
        g = np.clip(np.floor(t_norm * (c - 1)).astype(np.int64), 0, c - 1)
        sign = np.where(p > 0, 1.0, -1.0).astype(np.float32)
        np.add.at(rep, (g, y, x), sign)

        max_abs = np.abs(rep).max()
        if max_abs > 0:
            rep /= max_abs
        return torch.from_numpy(rep)
