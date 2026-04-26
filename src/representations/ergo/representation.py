"""
ERGO-12 Event Representation
论文: From Chaos Comes Order: Ordering Event Representations for Object Recognition and Detection
会议: ICCV 2023
官方仓库: https://github.com/uzh-rpg/event_representation_study

选择理由:
  ERGO-12 是该论文通过 Bayesian 优化发现的最优 12 通道 dense event representation，
  在 classification 和 detection 任务上均有使用，适合 N-Caltech101 分类任务。
  输出形状 (12, H, W)，可直接接入 ResNet18 classifier。

实现说明:
  - 本实现为独立 wrapper，基于原论文代码逻辑改写
  - 不依赖 torch_scatter，使用 numpy 的 np.add.at / np.maximum.at 实现 scatter 聚合
  - 时间窗口采用 SBN（Stacking Based on Number）策略：按事件数量划分为 7 个窗口
  - 7 个时间窗口定义（假设事件已按时间排序，N = 总事件数）:
      Window 0: 全部事件 [0 : N]
      Window 1: 前 1/3  [0 : N//3]
      Window 2: 中 1/3  [N//3 : 2*N//3]
      Window 3: 后 1/3  [2*N//3 : N]
      Window 4: 后 1/2  [N//2 : N]
      Window 5: 后 1/4  [3*N//4 : N]
      Window 6: 后 1/8  [7*N//8 : N]
  - 12 通道配置（window_idx, function, aggregation），来自原论文优化结果:
      Ch  0: (w=0, polarity,       variance)
      Ch  1: (w=3, timestamp_neg,  variance)
      Ch  2: (w=2, count_neg,      mean    )
      Ch  3: (w=6, polarity,       sum     )
      Ch  4: (w=5, count_pos,      mean    )
      Ch  5: (w=6, count,          sum     )
      Ch  6: (w=2, timestamp_pos,  mean    )
      Ch  7: (w=5, count_neg,      mean    )
      Ch  8: (w=1, timestamp_neg,  max     )
      Ch  9: (w=0, timestamp_pos,  max     )
      Ch 10: (w=4, timestamp,      max     )
      Ch 11: (w=1, count,          mean    )
"""

import numpy as np
import torch
import sys
import os

# 确保能找到 base.py 和 registry.py
_REPR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPR_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_REPR_DIR))

from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation

# ──────────────────────────────────────────────────────────────────────────────
# ERGO-12 固定配置（来自论文 Bayesian 优化结果）
# ──────────────────────────────────────────────────────────────────────────────
N_CHANNELS = 12

WINDOW_INDEXES = [0, 3, 2, 6, 5, 6, 2, 5, 1, 0, 4, 1]
FUNCTIONS = [
    "polarity",       # ch 0
    "timestamp_neg",  # ch 1
    "count_neg",      # ch 2
    "polarity",       # ch 3
    "count_pos",      # ch 4
    "count",          # ch 5
    "timestamp_pos",  # ch 6
    "count_neg",      # ch 7
    "timestamp_neg",  # ch 8
    "timestamp_pos",  # ch 9
    "timestamp",      # ch 10
    "count",          # ch 11
]
AGGREGATIONS = [
    "variance",  # ch 0
    "variance",  # ch 1
    "mean",      # ch 2
    "sum",       # ch 3
    "mean",      # ch 4
    "sum",       # ch 5
    "mean",      # ch 6
    "mean",      # ch 7
    "max",       # ch 8
    "max",       # ch 9
    "max",       # ch 10
    "mean",      # ch 11
]


def _make_windows(x, y, t_norm, p_pm1, N):
    """
    按 SBN 策略将事件划分为 7 个时间窗口。
    p_pm1: polarity 已转为 {-1, +1}
    返回长度为 7 的列表，每个元素为 (x, y, t_norm, p_pm1) 的 slice。
    """
    if N == 0:
        empty = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int32)
        return [(empty_i, empty_i, empty, empty)] * 7

    t1 = N // 3
    t2 = 2 * N // 3
    h  = N // 2
    q  = 3 * N // 4   # start of last-quarter slice
    e  = 7 * N // 8   # start of last-eighth slice

    slices_def = [
        (0,  N),   # window 0: all
        (0,  t1),  # window 1: first 1/3
        (t1, t2),  # window 2: mid 1/3
        (t2, N),   # window 3: last 1/3
        (h,  N),   # window 4: last 1/2
        (q,  N),   # window 5: last 1/4
        (e,  N),   # window 6: last 1/8
    ]
    windows = []
    for (s, e_) in slices_def:
        windows.append((x[s:e_], y[s:e_], t_norm[s:e_], p_pm1[s:e_]))
    return windows


def _scatter_aggregate(values, yi, xi, H, W, agg):
    """
    Scatter-aggregate values onto a (H, W) surface.
    使用 numpy 的 np.add.at / np.maximum.at 实现。
    """
    surf = np.zeros((H, W), dtype=np.float64)

    if len(values) == 0:
        return surf

    if agg == "sum":
        np.add.at(surf, (yi, xi), values)

    elif agg == "mean":
        count = np.zeros((H, W), dtype=np.float64)
        np.add.at(surf, (yi, xi), values)
        np.add.at(count, (yi, xi), 1.0)
        mask = count > 0
        surf[mask] /= count[mask]

    elif agg == "max":
        surf.fill(-np.inf)
        np.maximum.at(surf, (yi, xi), values)
        surf[surf == -np.inf] = 0.0

    elif agg == "variance":
        # Var = E[x^2] - (E[x])^2
        sum_sq  = np.zeros((H, W), dtype=np.float64)
        sum_v   = np.zeros((H, W), dtype=np.float64)
        count   = np.zeros((H, W), dtype=np.float64)
        np.add.at(sum_sq, (yi, xi), values ** 2)
        np.add.at(sum_v,  (yi, xi), values)
        np.add.at(count,  (yi, xi), 1.0)
        mask = count > 0
        e_x2 = np.zeros((H, W), dtype=np.float64)
        e_x  = np.zeros((H, W), dtype=np.float64)
        e_x2[mask] = sum_sq[mask] / count[mask]
        e_x[mask]  = sum_v[mask]  / count[mask]
        surf = e_x2 - e_x ** 2

    return surf


def _get_feature_values(fn_name, t_norm, p_pm1):
    """
    根据 function 名称提取对应的特征值数组。
    p_pm1 ∈ {-1, +1}
    """
    pos_mask = p_pm1 > 0   # positive events
    neg_mask = p_pm1 < 0   # negative events

    if fn_name == "polarity":
        return p_pm1.astype(np.float64)

    elif fn_name == "timestamp":
        return t_norm

    elif fn_name == "timestamp_pos":
        vals = np.zeros_like(t_norm)
        vals[pos_mask] = t_norm[pos_mask]
        return vals

    elif fn_name == "timestamp_neg":
        vals = np.zeros_like(t_norm)
        vals[neg_mask] = t_norm[neg_mask]
        return vals

    elif fn_name == "count":
        return np.ones(len(t_norm), dtype=np.float64)

    elif fn_name == "count_pos":
        return pos_mask.astype(np.float64)

    elif fn_name == "count_neg":
        return neg_mask.astype(np.float64)

    else:
        raise ValueError(f"Unknown function: {fn_name}")


@register_representation("ergo")
class EventOrderingRepresentation(BaseRepresentation):
    """
    ERGO-12 — 来自 'From Chaos Comes Order' (ICCV 2023)。
    输出: torch.FloatTensor, shape = (12, height, width)
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)
        self.height     = int(config.get("height",     180))
        self.width      = int(config.get("width",      240))
        self.max_events = int(config.get("max_events", 50000))
        self.n_channels = N_CHANNELS

    def build(self, events):
        """
        输入:
            events: structured numpy array，字段 (x, y, t, p)
                    或 ndarray shape (N, 4) — columns [x, y, t, p]
        输出:
            torch.FloatTensor, shape = (12, height, width)
        """
        H, W = self.height, self.width

        # ── 解析 structured array ────────────────────────────────────────────
        if events is None or (hasattr(events, "__len__") and len(events) == 0):
            return torch.zeros((N_CHANNELS, H, W), dtype=torch.float32)

        if events.dtype.names is not None:
            # Tonic structured array
            x_raw = events["x"].astype(np.int32)
            y_raw = events["y"].astype(np.int32)
            t_raw = events["t"].astype(np.float64)
            p_raw = events["p"].astype(np.int32)
        else:
            arr   = np.asarray(events)
            x_raw = arr[:, 0].astype(np.int32)
            y_raw = arr[:, 1].astype(np.int32)
            t_raw = arr[:, 2].astype(np.float64)
            p_raw = arr[:, 3].astype(np.int32)

        N = len(x_raw)
        if N == 0:
            return torch.zeros((N_CHANNELS, H, W), dtype=torch.float32)

        # ── 截断事件数 ──────────────────────────────────────────────────────
        if N > self.max_events:
            # 保留最新的 max_events 个事件（按时间从小到大排列）
            x_raw = x_raw[-self.max_events:]
            y_raw = y_raw[-self.max_events:]
            t_raw = t_raw[-self.max_events:]
            p_raw = p_raw[-self.max_events:]
            N = self.max_events

        # ── 坐标裁剪（防止越界）─────────────────────────────────────────────
        x_raw = np.clip(x_raw, 0, W - 1)
        y_raw = np.clip(y_raw, 0, H - 1)

        # ── 时间戳归一化 [0, 1] ──────────────────────────────────────────────
        t_min, t_max = t_raw.min(), t_raw.max()
        if t_max > t_min:
            t_norm = (t_raw - t_min) / (t_max - t_min)
        else:
            # 所有时间戳相同，统一设为 0.5
            t_norm = np.full(N, 0.5, dtype=np.float64)

        # ── 极性转为 {-1, +1} ─────────────────────────────────────────────
        # Tonic NCALTECH101 返回 p ∈ {0, 1}
        p_pm1 = np.where(p_raw > 0, 1.0, -1.0).astype(np.float64)

        # ── 划分 7 个时间窗口 ────────────────────────────────────────────────
        windows = _make_windows(x_raw, y_raw, t_norm, p_pm1, N)

        # ── 构建 12 个通道 ────────────────────────────────────────────────────
        channels = []
        for ch in range(N_CHANNELS):
            win_idx = WINDOW_INDEXES[ch]
            fn_name = FUNCTIONS[ch]
            agg     = AGGREGATIONS[ch]

            xi, yi, t_w, p_w = windows[win_idx]

            if len(xi) == 0:
                # 该窗口为空，填零
                channels.append(np.zeros((H, W), dtype=np.float64))
                continue

            feat = _get_feature_values(fn_name, t_w, p_w)
            surf = _scatter_aggregate(feat, yi, xi, H, W, agg)
            channels.append(surf)

        # ── 堆叠并转为 tensor ─────────────────────────────────────────────────
        arr = np.stack(channels, axis=0).astype(np.float32)  # (12, H, W)
        tensor = torch.from_numpy(arr)

        # ── 全局归一化（channel-wise min-max）以稳定训练 ─────────────────────
        for c in range(N_CHANNELS):
            ch_min = tensor[c].min()
            ch_max = tensor[c].max()
            if ch_max > ch_min:
                tensor[c] = (tensor[c] - ch_min) / (ch_max - ch_min)

        return tensor  # shape: (12, H, W)
