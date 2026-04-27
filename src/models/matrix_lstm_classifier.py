"""
End-to-end Matrix-LSTM classifier（DataLoader 并行版）

关键优化：
  CPU 分组（lexsort/unique/split）移到 Dataset.__getitem__()，
  由 4 个 DataLoader worker 并行执行。
  model.forward() 里无任何 Python 循环，只有纯 GPU 操作。

输入格式（由 matrix_lstm_collate 生成）：
  seqs        : (total_N_pix, MAX_EV_RF, 2)   float32  GPU
  lens        : (total_N_pix,)                int64
  gpix        : (total_N_pix,)                int64    GPU  全局像素 id
  sample_npix : (B,)                          int64    每个样本的活跃像素数
"""

import torch, torch.nn as nn
from src.representations.matrix_lstm.representation import MatrixLSTMSurface
from src.models.classifier import EventClassifier

class MatrixLSTMClassifier(nn.Module):

    def __init__(self, height=180, width=240, hidden_size=16,
                 num_classes=101, max_events_per_rf=32):
        super().__init__()
        self.height      = height
        self.width       = width
        self.hidden_size = hidden_size
        self.surface     = MatrixLSTMSurface(input_size=2, hidden_size=hidden_size)
        self.classifier  = EventClassifier(in_channels=hidden_size, num_classes=num_classes)

    def forward(self, seqs, lens, gpix, sample_npix):
        """
        无 Python 循环，纯 GPU 操作。
        seqs   : (total_N_pix, MAX_EV_RF, 2)
        lens   : (total_N_pix,)
        gpix   : (total_N_pix,)  全局像素 id = sample_i * H * W + local_pix
        sample_npix: (B,)
        """
        B      = sample_npix.shape[0]
        device = seqs.device
        H, W, C = self.height, self.width, self.hidden_size

        if seqs.shape[0] == 0:
            return self.classifier(torch.zeros(B, C, H, W, device=device))

        # 单次 LSTM 调用（所有样本所有像素合并）
        hidden = self.surface(seqs, lens)                       # (total_N_pix, C)

        # Scatter → (B, C, H, W)
        surf = torch.zeros(B * H * W, C, device=device).scatter(
            0, gpix.unsqueeze(1).expand(-1, C), hidden
        )
        surf = surf.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return self.classifier(surf)
