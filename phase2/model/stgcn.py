"""
stgcn.py
--------
Lightweight Spatial-Temporal Graph Convolutional Network.
CPU-optimised: 3 layers, 64 channels max (vs 256 in original ST-GCN).

Input:  (B, C, T, 2, 17)  — batch, channels, time, persons, joints
Output: (B, 4)             — logits for 4 classes

Graph: COCO 17-joint skeleton connectivity.
Each ST-GCN layer does spatial graph conv across joints,
then temporal conv across time frames.

The key is the graph convolution learns which joints to
attend to together — it discovers "wrist moving fast toward
the other person's face" without being told to look at those joints.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_JOINTS = 17
N_CLASSES = 4

# COCO 17 skeleton edges (bone connections)
# The graph adjacency captures spatial body structure
SKELETON_EDGES = [
    (0, 1), (0, 2),        # nose → eyes
    (1, 3), (2, 4),        # eyes → ears
    (0, 5), (0, 6),        # nose → shoulders (through neck implicitly)
    (5, 6),                # shoulder–shoulder
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (5, 11), (6, 12),      # torso
    (11, 12),              # hip–hip
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
]


def build_adjacency(strategy: str = "spatial") -> torch.Tensor:
    """
    Build normalized adjacency matrix for the COCO 17-joint skeleton.

    strategy:
      'spatial'  — binary connectivity (bone exists = 1)
      'distance' — distance-partitioned (self, neighbor, further)

    Returns: (1, 17, 17) normalized adj matrix
    """
    A = np.zeros((N_JOINTS, N_JOINTS), dtype=np.float32)
    # Self-connections
    np.fill_diagonal(A, 1)
    # Bone connections (bidirectional)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1
        A[j, i] = 1

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = np.diag(A.sum(axis=1) ** -0.5)
    A_norm = D @ A @ D

    return torch.tensor(A_norm, dtype=torch.float32).unsqueeze(0)  # (1, 17, 17)


class SpatialGraphConv(nn.Module):
    """
    Single spatial graph convolution layer.
    Mixes joint features according to the skeleton adjacency.
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)   # (1, 17, 17)
        # Learnable edge importance — lets the model up/down-weight specific joints
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, V, T) — batch, channels, joints, time
        A_eff = self.A * self.edge_importance   # learned importance
        # Graph conv: x @ A — mix joint features
        x = torch.einsum("bcvt,kvj->bcjt", x, A_eff)

        x = self.conv(x)
        return self.bn(x)


class STGCNBlock(nn.Module):
    """
    One ST-GCN block: spatial graph conv → temporal conv → residual.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)

        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1),
                      padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection: match channel/stride if needed
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        # ST-GCN convention: swap to (B, C, V, T) for spatial conv
        x_spatial = x.permute(0, 1, 3, 2)           # (B, C, V, T)
        out = self.gcn(x_spatial)                    # (B, C', V, T)
        out = out.permute(0, 1, 3, 2)               # (B, C', T, V)
        out = self.tcn(out)                          # temporal conv
        out = self.relu(out + self.residual(x))
        return self.dropout(out)


class LightSTGCN(nn.Module):
    """
    Lightweight ST-GCN for CPU inference.

    Architecture:
      input_bn → 3 × STGCNBlock → global_avg_pool → classifier

    Channel progression: C_in → 32 → 64 → 64
    Compared to original ST-GCN (64→128→256), this is ~4× fewer params.
    On CPU (i5/i7) expect ~15-25ms per batch=1 forward pass.

    Input shape: (B, C, T, 2, 17)
      B = batch, C = feature channels, T = 30 frames,
      2 = persons, 17 = joints

    The 2-person dimension is handled by treating each person as a
    separate "graph instance" and concatenating their features before
    the classifier — this lets the model learn cross-person patterns.
    """

    def __init__(
        self,
        in_channels: int = 19,   # from FeatureConstructor.n_channels
        num_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()

        A = build_adjacency()  # (1, 17, 17)

        self.input_bn = nn.BatchNorm1d(in_channels * N_JOINTS * 2)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 32, A, temporal_kernel=9, dropout=dropout),
            STGCNBlock(32, 64, A, temporal_kernel=9, dropout=dropout),
            STGCNBlock(64, 64, A, temporal_kernel=5, dropout=dropout),
        ])

        # Global average pool over time and joints, then classify
        # After pool: (B, 64, 2) — 64 channels per person
        # Concat persons: (B, 128)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, 2, 17)

        Returns:
            logits: (B, num_classes)
        """
        B, C, T, P, V = x.shape  # P=2 persons, V=17 joints

        # Flatten for input BN: (B, C*V*P)
        x_flat = x.permute(0, 1, 4, 3, 2).reshape(B, C * V * P, T)
        # BN over feature dim: (B, C*V*P, T) — use BN1d on the spatial dim
        x_flat_bn = self.input_bn(x_flat.reshape(B, C * V * P, -1).mean(-1, keepdim=True).expand_as(x_flat.reshape(B, C * V * P, T)))

        # Reshape back: process each person separately through GCN blocks
        # Merge persons into batch: (B*P, C, T, V)
        x_merged = x.permute(0, 3, 1, 2, 4).reshape(B * P, C, T, V)

        for block in self.blocks:
            x_merged = block(x_merged)

        # x_merged: (B*P, 64, T', V)
        # Global average pool over time and joints
        x_pooled = x_merged.mean(dim=[2, 3])   # (B*P, 64)

        # Separate persons back: (B, P, 64)
        x_persons = x_pooled.reshape(B, P, -1)

        # Concatenate both persons' features: (B, 128)
        x_concat = x_persons.reshape(B, -1)

        return self.classifier(x_concat)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
