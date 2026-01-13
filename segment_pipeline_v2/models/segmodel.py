import torch
from torch import nn
from torch.nn import functional as F


class SegModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        if isinstance(logits, (tuple, list)):
            logits = logits[-1]
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
