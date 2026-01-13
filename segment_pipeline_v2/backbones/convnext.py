from typing import List, Optional

import torch
from torch import nn, Tensor

from layers import DropPath
from .utils import load_state_dict


class LayerNorm(nn.Module):
    """Channel first layer norm"""

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Module):
    def __init__(self, dim, dpr=0.0, init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True) if init_value > 0 else None
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)
        x = identity + self.drop_path(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, c1, c2, k, s):
        super().__init__(
            nn.Conv2d(c1, c2, k, s),
            LayerNorm(c2),
        )


class Downsample(nn.Sequential):
    def __init__(self, c1, c2, k, s):
        super().__init__(
            LayerNorm(c1),
            nn.Conv2d(c1, c2, k, s),
        )


convnext_settings = {
    "T": [[3, 3, 9, 3], [96, 192, 384, 768], 0.0],
    "S": [[3, 3, 27, 3], [96, 192, 384, 768], 0.0],
    "B": [[3, 3, 27, 3], [128, 256, 512, 1024], 0.0],
}


class ConvNeXt(nn.Module):
    def __init__(self, model_name: str = "T") -> None:
        super().__init__()
        assert model_name in convnext_settings.keys(), (
            f"ConvNeXt model name should be in {list(convnext_settings.keys())}"
        )
        depths, embed_dims, drop_path_rate = convnext_settings[model_name]
        self.channels = embed_dims
        self.out_channels = embed_dims

        self.downsample_layers = nn.ModuleList(
            [
                Stem(3, embed_dims[0], 4, 4),
                *[Downsample(embed_dims[i], embed_dims[i + 1], 2, 2) for i in range(3)],
            ]
        )

        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(*[Block(embed_dims[i], dpr[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        for i in range(4):
            self.add_module(f"norm{i}", LayerNorm(embed_dims[i]))

    def forward(self, x: Tensor):
        outs = []

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f"norm{i}")
            outs.append(norm_layer(x))
        return outs


class TorchvisionConvNeXtTBackbone(nn.Module):
    def __init__(self, pretrained: bool, weights_path: Optional[str]) -> None:
        super().__init__()
        try:
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        except Exception as exc:
            raise RuntimeError("torchvision이 필요합니다. torchvision 설치를 확인하세요.") from exc
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = convnext_tiny(weights=weights)
        load_state_dict(self.model, weights_path)
        self.out_channels = [96, 192, 384, 768]
        self.channels = self.out_channels

    def forward(self, x: Tensor) -> List[Tensor]:
        outs = []
        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if idx in (1, 3, 5, 7):
                outs.append(x)
        return outs
