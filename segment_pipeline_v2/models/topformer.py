from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from backbones import ConvBNReLU


class TopFormerHead(nn.Module):
    def __init__(self, in_channels: List[int], num_classes: int, head_channels: int) -> None:
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, head_channels, 1) for c in in_channels])
        self.fpn_convs = nn.ModuleList([ConvBNReLU(head_channels, head_channels, 3, 1, 1) for _ in in_channels])
        self.cls = nn.Conv2d(head_channels, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        target_size = outs[0].shape[-2:]
        fused = sum(F.interpolate(o, size=target_size, mode="bilinear", align_corners=False) for o in outs)
        return self.cls(fused)
