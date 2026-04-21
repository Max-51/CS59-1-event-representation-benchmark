import numpy as np
import torch
from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation

@register_representation("est")
class ESTRepresentation(BaseRepresentation):
    def __init__(self, config):
        super().__init__(config)
        self.num_bins = config.get("num_bins", 5)
        self.height   = config.get("height", 180)
        self.width    = config.get("width", 240)

    def build(self, events):
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2]
        p = events[:, 3].astype(np.int32)

        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min) * (self.num_bins - 1)
        else:
            t_norm = np.zeros_like(t)
        bins = np.floor(t_norm).astype(np.int32).clip(0, self.num_bins - 1)

        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, bins, p = x[valid], y[valid], bins[valid], p[valid]

        voxel = np.zeros((2 * self.num_bins, self.height, self.width), dtype=np.float32)
        np.add.at(voxel, (p * self.num_bins + bins, y, x), 1.0)

        voxel_max = voxel.max()
        if voxel_max > 0:
            voxel /= voxel_max

        return torch.from_numpy(voxel)
