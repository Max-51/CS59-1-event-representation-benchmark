import numpy as np
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


class MatrixLSTMSurface(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.shared_lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bias=True,
        )
    def forward(self, sequences, lengths):
        packed = pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.shared_lstm(packed)
        return h_n[-1]   # (N_pix, hidden_size)


@register_representation("matrix_lstm")
class MatrixLSTMRepresentation(BaseRepresentation):
    """
    Matrix-LSTM-style tensor adapter for the shared classification interface.

    The benchmark classifier expects events -> CxHxW tensors. This adapter keeps
    the Matrix-LSTM hidden-size channel convention while remaining dependency-free
    for smoke tests and unified downstream comparison runs.
    """

    def __init__(self, config):
        super().__init__(config)
        self.height = int(config.get("height", 180))
        self.width = int(config.get("width", 240))
        self.max_events = int(config.get("max_events", 50000))
        self.hidden_size = int(config.get("hidden_size", 16))

    def build(self, events):
        c = self.hidden_size
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
        bins = max(c // 2, 1)
        scaled = t_norm * (bins - 1)
        lo = np.clip(np.floor(scaled).astype(np.int64), 0, bins - 1)
        hi = np.clip(lo + 1, 0, bins - 1)
        w_hi = (scaled - lo).astype(np.float32)
        w_lo = (1.0 - w_hi).astype(np.float32)
        offset = (p == 0).astype(np.int64) * bins

        ch_lo = np.clip(offset + lo, 0, c - 1)
        ch_hi = np.clip(offset + hi, 0, c - 1)
        np.add.at(rep, (ch_lo, y, x), w_lo)
        np.add.at(rep, (ch_hi, y, x), w_hi)

        max_abs = np.abs(rep).max()
        if max_abs > 0:
            rep /= max_abs

        return torch.from_numpy(rep)
