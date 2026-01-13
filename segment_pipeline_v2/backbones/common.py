import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
