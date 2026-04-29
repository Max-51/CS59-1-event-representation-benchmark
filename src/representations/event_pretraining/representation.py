"""
Event Camera Data Pre-training (ICCV 2023) — Representation

论文核心思想：
  用正/负极性分离的 2-channel event image 作为输入，
  通过对比学习预训练 encoder，再迁移到下游分类任务。

本文件实现：
  事件 → 2-channel polarity accumulation image (2, H, W)
  Channel 0: 正极性事件计数 (p=1)，归一化
  Channel 1: 负极性事件计数 (p=0)，归一化

experiment_type: small_scale_simclr_pretrain
  - 在 N-Caltech101 train split 上做 SimCLR 自监督预训练
  - 再用有标签数据 finetune
  - 这是论文预训练思想的小规模适配，非原文完整复现
"""

import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


@register_representation("event_pretraining")
class EventPretrainingRepresentation(BaseRepresentation):
    """
    2-channel polarity-accumulation event frame.
    输出 shape: (2, height, width)，默认 (2, 224, 224)
    """

    def __init__(self, config=None):
        super().__init__(config)
        cfg = config or {}
        self.height     = int(cfg.get("height",     224))
        self.width      = int(cfg.get("width",      224))
        self.max_events = int(cfg.get("max_events", 50000))

    @property
    def output_channels(self):
        return 2

    def build(self, events):
        """
        Parameters
        ----------
        events : numpy structured array, fields: x, y, t, p

        Returns
        -------
        torch.Tensor  shape (2, height, width)  dtype float32
        """
        if events is None or len(events) == 0:
            return torch.zeros(2, self.height, self.width, dtype=torch.float32)

        x = events["x"].astype(np.int64)
        y = events["y"].astype(np.int64)
        p = events["p"].astype(np.int64)

        # 超出 max_events 则均匀采样
        if len(x) > self.max_events:
            idx = np.random.choice(len(x), self.max_events, replace=False)
            x, y, p = x[idx], y[idx], p[idx]

        # 坐标裁剪
        x = np.clip(x, 0, self.width  - 1)
        y = np.clip(y, 0, self.height - 1)

        frame = np.zeros((2, self.height, self.width), dtype=np.float32)
        pos = (p == 1)
        neg = (p == 0)
        np.add.at(frame[0], (y[pos], x[pos]), 1.0)
        np.add.at(frame[1], (y[neg], x[neg]), 1.0)

        # 每个通道独立归一化到 [0, 1]
        for c in range(2):
            m = frame[c].max()
            if m > 0:
                frame[c] /= m

        return torch.from_numpy(frame)
