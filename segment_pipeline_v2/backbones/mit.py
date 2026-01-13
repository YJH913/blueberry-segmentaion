from typing import List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from layers import DropPath
from .utils import load_state_dict


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.head, c // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(b, c, H, W)
            x = self.sr(x).reshape(b, c, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(b, -1, 2, self.head, c // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size // 2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


mit_settings = {
    "B0": [[32, 64, 160, 256], [2, 2, 2, 2]],
    "B1": [[64, 128, 320, 512], [2, 2, 2, 2]],
    "B2": [[64, 128, 320, 512], [3, 4, 6, 3]],
    "B3": [[64, 128, 320, 512], [3, 4, 18, 3]],
    "B4": [[64, 128, 320, 512], [3, 8, 27, 3]],
    "B5": [[64, 128, 320, 512], [3, 6, 40, 3]],
}


class MiT(nn.Module):
    def __init__(self, model_name: str = "B0"):
        super().__init__()
        assert model_name in mit_settings.keys(), f"MiT model name should be in {list(mit_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.out_channels = embed_dims

        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) -> List[Tensor]:
        b = x.shape[0]
        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x1 = self.norm1(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)

        x, h, w = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, h, w)
        x2 = self.norm2(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)

        x, h, w = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, h, w)
        x3 = self.norm3(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)

        x, h, w = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, h, w)
        x4 = self.norm4(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)

        return [x1, x2, x3, x4]


class TimmMiTB0Backbone(nn.Module):
    def __init__(self, pretrained: bool, weights_path: Optional[str]) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("MiT-B0 사전학습은 timm이 필요합니다.") from exc
        self.model = timm.create_model("mit_b0", pretrained=pretrained, features_only=True)
        load_state_dict(self.model, weights_path)
        self.out_channels = list(self.model.feature_info.channels())
        self.channels = self.out_channels

    def forward(self, x: Tensor) -> List[Tensor]:
        return list(self.model(x))
