"""
EST End-to-End Classifier
=========================

Faithful reproduction of the learnable representation from:
    "End-to-End Learning of Representations for Asynchronous
     Event-Based Data" (Gehrig et al., ICCV 2019)
    https://github.com/uzh-rpg/rpg_event_representation_learning

Key difference from EST adaptation (est/representation.py):
    - QuantizationLayer is a TRAINABLE nn.Module (small MLP)
    - Gradient flows from classification loss → MLP → representation
    - The representation is optimised jointly with the backbone
    - This file: est_e2e.py (end-to-end)
    - Previous file: est/representation.py (fixed preprocessing, adaptation)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34


# ══════════════════════════════════════════════════════════════════
# QuantizationLayer
# ══════════════════════════════════════════════════════════════════
class QuantizationLayer(nn.Module):
    """
    Learnable event-to-voxel quantization layer (original EST).

    For each event, maps its normalised timestamp t ∈ [0,1] to a
    soft weight vector over num_bins temporal bins via a tiny MLP:

        t  →  Linear(1→16)  →  LeakyReLU(0.1)
           →  Linear(16→num_bins)  →  ReLU  →  q ∈ R^{num_bins}_{≥0}

    These weights are accumulated into a voxel grid via a
    differentiable scatter-add, so gradients flow back through
    q all the way to the MLP parameters.

    Args:
        num_bins (int): Number of temporal bins.
    """
    def __init__(self, num_bins: int):
        super().__init__()
        self.num_bins = num_bins
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(16, num_bins),
            nn.ReLU(),
        )

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_norm: (N,) normalised timestamps in [0, 1]
        Returns:
            q: (N, num_bins)  non-negative soft bin weights
        """
        return self.mlp(t_norm.unsqueeze(-1))   # (N, num_bins)


# ══════════════════════════════════════════════════════════════════
# ESTEndToEndClassifier
# ══════════════════════════════════════════════════════════════════
class ESTEndToEndClassifier(nn.Module):
    """
    End-to-end EST event classification model.

    Forward pipeline
    ────────────────
    raw events (B, N, 5)                        [x, y, t, p, valid]
        │
        ├─ timestamp normalisation (per sample)
        │
        ├─ QuantizationLayer  (learnable MLP)   t → q ∈ R^K
        │
        ├─ zero-out padding events              q *= valid_mask
        │
        ├─ differentiable scatter-add           q → voxel (C, H, W)
        │      C = 2 * num_bins
        │      gradients flow through scatter → q → MLP
        │
        ├─ L-inf normalisation per sample
        │
        ├─ [optional] Gaussian noise injection  (robustness test)
        │
        └─ ResNet18/34 backbone                 → logits (B, num_classes)

    Input tensor column layout
    ──────────────────────────
    col 0: x           integer pixel coordinate
    col 1: y           integer pixel coordinate
    col 2: t           raw timestamp (any unit; normalised internally)
    col 3: p           polarity  0 = negative,  1 = positive
    col 4: valid_mask  1 = real event,  0 = padding

    Channel layout in voxel grid
    ─────────────────────────────
    p=1 (positive)  →  channels   0 ..   num_bins-1
    p=0 (negative)  →  channels num_bins .. 2*num_bins-1

    Args:
        num_bins   (int): Temporal bins. Default: 9  → 18 channels.
        height     (int): Sensor height. Default: 180.
        width      (int): Sensor width.  Default: 240.
        num_classes(int): Output classes. Default: 101.
        backbone   (str): 'resnet18' (benchmark default) or
                          'resnet34' (original paper). Default: 'resnet18'.
    """

    def __init__(
        self,
        num_bins:    int = 9,
        height:      int = 180,
        width:       int = 240,
        num_classes: int = 101,
        backbone:    str = "resnet18",
    ):
        super().__init__()
        self.num_bins      = num_bins
        self.height        = height
        self.width         = width
        self.C             = 2 * num_bins       # total voxel channels
        self.backbone_name = backbone

        # ── Learnable quantization layer ──────────────────────────
        self.quantization = QuantizationLayer(num_bins)

        # ── Backbone ──────────────────────────────────────────────
        if backbone == "resnet18":
            net = resnet18(weights=None)
        elif backbone == "resnet34":
            net = resnet34(weights=None)
        else:
            raise ValueError(
                f"backbone must be 'resnet18' or 'resnet34', got '{backbone}'"
            )

        # First conv: 3 → 2*num_bins channels
        net.conv1 = nn.Conv2d(
            self.C, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # FC: → num_classes
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.backbone = net

    # ──────────────────────────────────────────────────────────────
    def _build_voxel_single(
        self,
        x: torch.Tensor,   # (N,) int64  valid events only
        y: torch.Tensor,   # (N,) int64
        p: torch.Tensor,   # (N,) int64  in {0, 1}
        q: torch.Tensor,   # (N, num_bins)  MLP output, non-negative
    ) -> torch.Tensor:
        """
        Differentiable voxel construction for one sample.

        Gradients propagate through scatter_add → q → QuantizationLayer MLP.
        """
        H, W, C, K = self.height, self.width, self.C, self.num_bins
        device = q.device
        N = x.shape[0]

        if N == 0:
            return torch.zeros(C, H, W, device=device, dtype=q.dtype)

        # Safety clamp (guards against rare out-of-bounds coordinates)
        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)
        p = p.clamp(0, 1)

        # Channel offset:  p=1 → 0,  p=0 → num_bins
        pol_offset = (p == 0).long() * K              # (N,)

        # For each event × bin pair: channel index and flat position
        bin_range = torch.arange(K, device=device)                   # (K,)
        ch  = pol_offset.unsqueeze(1) + bin_range.unsqueeze(0)       # (N, K)
        x_e = x.unsqueeze(1).expand(-1, K)                           # (N, K)
        y_e = y.unsqueeze(1).expand(-1, K)                           # (N, K)

        flat_idx = (ch * H * W + y_e * W + x_e).reshape(-1)          # (N*K,)
        q_flat   = q.reshape(-1)                                       # (N*K,)

        # Differentiable scatter-add  ← gradients flow through here
        voxel_flat = torch.zeros(
            C * H * W, device=device, dtype=q.dtype
        )
        voxel_flat.scatter_add_(0, flat_idx, q_flat)
        voxel = voxel_flat.view(C, H, W)

        # L-inf normalisation (differentiable)
        v_max = voxel.abs().max()
        if v_max > 0:
            voxel = voxel / v_max

        return voxel

    # ──────────────────────────────────────────────────────────────
    def forward(
        self,
        events_batch: torch.Tensor,
        keep_ratio:   float = 1.0,
        noise_sigma:  float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            events_batch: (B, N, 5)  [x, y, t, p, valid_mask]
            keep_ratio:   sparsity test — fraction of events to keep.
                          Applied by randomly masking real events.
            noise_sigma:  noise test — std of Gaussian noise added to
                          the voxel grid after construction.
        Returns:
            logits: (B, num_classes)
        """
        B, N, _ = events_batch.shape
        device  = events_batch.device

        x_all  = events_batch[:, :, 0].long()    # (B, N)
        y_all  = events_batch[:, :, 1].long()    # (B, N)
        t_all  = events_batch[:, :, 2].float()   # (B, N)
        p_all  = events_batch[:, :, 3].long()    # (B, N)
        vmask  = events_batch[:, :, 4].float()   # (B, N) 1=real 0=pad

        # ── Optional sparsity: randomly drop real events ───────────
        if keep_ratio < 1.0:
            drop  = torch.bernoulli(
                torch.full((B, N), keep_ratio, device=device)
            )
            vmask = vmask * drop   # only drop among real events

        # ── Timestamp normalisation per sample ────────────────────
        # min/max computed over ALL events in each sample row.
        # Padded events are copies of real events, so their t is valid.
        t_min = t_all.min(dim=1, keepdim=True)[0]   # (B, 1)
        t_max = t_all.max(dim=1, keepdim=True)[0]   # (B, 1)
        t_norm = (t_all - t_min) / (t_max - t_min + 1e-6)   # (B, N)
        t_norm = t_norm.clamp(0.0, 1.0)

        # ── QuantizationLayer: one batched MLP call ───────────────
        # Process all B*N timestamps simultaneously for GPU efficiency
        q_flat = self.quantization.mlp(
            t_norm.reshape(B * N, 1)
        )                                            # (B*N, num_bins)
        q_all  = q_flat.view(B, N, self.num_bins)   # (B, N, num_bins)

        # Zero out padded events BEFORE scatter (no gradient through padding)
        q_all = q_all * vmask.unsqueeze(-1)          # (B, N, num_bins)

        # ── Build voxel grid per sample ───────────────────────────
        voxels = []
        for b in range(B):
            mask_b = vmask[b].bool()                 # (N,)
            voxels.append(
                self._build_voxel_single(
                    x_all[b][mask_b],
                    y_all[b][mask_b],
                    p_all[b][mask_b],
                    q_all[b][mask_b],
                )
            )
        voxel_batch = torch.stack(voxels)            # (B, C, H, W)

        # ── Optional noise injection ───────────────────────────────
        if noise_sigma > 0.0:
            voxel_batch = (
                voxel_batch + torch.randn_like(voxel_batch) * noise_sigma
            )

        # ── ResNet backbone ───────────────────────────────────────
        return self.backbone(voxel_batch)
