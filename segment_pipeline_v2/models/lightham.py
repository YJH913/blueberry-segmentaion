from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from backbones import ConvBNReLU


class LightHamHead(nn.Module):
    def __init__(self, in_channels: List[int], num_classes: int, head_channels: int) -> None:
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, head_channels, 1) for c in in_channels])
        self.attn = nn.ModuleList([nn.Linear(head_channels, 1) for _ in in_channels])
        self.fuse = ConvBNReLU(head_channels, head_channels, 3, 1, 1)
        self.cls = nn.Conv2d(head_channels, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        feats = []
        scores = []
        target_size = features[0].shape[-2:]
        for feat, lateral, attn in zip(features, self.laterals, self.attn):
            x = lateral(feat)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            feats.append(x)
            pooled = x.mean(dim=(2, 3))
            scores.append(attn(pooled))
        weights = torch.softmax(torch.cat(scores, dim=1), dim=1)
        fused = sum(w[:, i : i + 1, None, None] * f for i, f in enumerate(feats))
        fused = self.fuse(fused)
        return self.cls(fused)
