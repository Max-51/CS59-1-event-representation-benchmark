
import sys
sys.path.insert(0, '/content/evrepsl_repo')

import numpy as np
import torch
from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation
from event_representations import events_to_EvRep


@register_representation("evrepsl")
class EvRepSLRepresentation(BaseRepresentation):
    """
    build() 返回 EvRep (3, H, W)。
    RepGen 在训练循环中以 batch 方式应用，输出 (5, H, W)。
    """
    def __init__(self, config):
        super().__init__(config)
        self.height = config.get("height", 180)
        self.width  = config.get("width",  240)
        self.max_events = config.get("max_events", 50000)

    def build(self, events):
        # events: structured numpy array with fields x, y, t, p (from Tonic)
        if hasattr(events, "dtype") and events.dtype.names:
            x = events["x"].astype(np.int32)
            y = events["y"].astype(np.int32)
            t = events["t"].astype(np.float64)
            p = events["p"].astype(np.int32)   # bool -> 0/1
        else:
            # fallback: plain (N, 4) array
            x = events[:, 0].astype(np.int32)
            y = events[:, 1].astype(np.int32)
            t = events[:, 2].astype(np.float64)
            p = events[:, 3].astype(np.int32)

        # 截断过长序列
        if len(x) > self.max_events:
            idx = np.random.choice(len(x), self.max_events, replace=False)
            idx.sort()
            x, y, t, p = x[idx], y[idx], t[idx], p[idx]

        # 过滤越界坐标
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, t, p = x[valid], y[valid], t[valid], p[valid]

        if len(x) == 0:
            return torch.zeros(3, self.height, self.width, dtype=torch.float32)

        # EvRep: (3, H, W)  [count, polarity_sum, temporal_std]
        ev_rep = events_to_EvRep(
            x, y, t, p,
            resolution=(self.width, self.height)
        ).astype(np.float32)

        # 逐通道归一化到 [0, 1]
        for i in range(3):
            ch_max = np.abs(ev_rep[i]).max()
            if ch_max > 0:
                ev_rep[i] /= ch_max

        return torch.from_numpy(ev_rep)  # (3, H, W)
