from typing import List

from torch import nn

from backbones import build_backbone
from .condnet import CondNetHead
from .lawin import LawinHead
from .lightham import LightHamHead
from .segmodel import SegModel
from .topformer import TopFormerHead


def build_head(name: str, in_channels: List[int], num_classes: int, head_channels: int) -> nn.Module:
    key = name.lower()
    if key == "condnet":
        return CondNetHead(in_channels, num_classes, head_channels)
    if key in ["light-ham", "lightham", "light_ham"]:
        return LightHamHead(in_channels, num_classes, head_channels)
    if key == "lawin":
        return LawinHead(in_channels, num_classes, head_channels)
    if key == "topformer":
        return TopFormerHead(in_channels, num_classes, head_channels)
    raise ValueError(f"지원하지 않는 HEAD입니다: {name}")


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    backbone = build_backbone(
        model_cfg["BACKBONE"],
        pretrained=bool(model_cfg.get("PRETRAINED", False)),
        weights_path=model_cfg.get("PRETRAINED_PATH"),
    )
    head_name = model_cfg.get("HEAD") or model_cfg.get("NAME")
    if not head_name:
        raise ValueError("MODEL.HEAD 또는 MODEL.NAME을 지정해야 합니다.")
    head_channels = model_cfg.get("HEAD_CHANNELS", 256)
    in_channels = getattr(backbone, "out_channels", None) or getattr(backbone, "channels", None)
    if in_channels is None:
        raise AttributeError("Backbone에 out_channels/channels 속성이 없습니다.")
    head = build_head(head_name, in_channels, num_classes, head_channels)
    return SegModel(backbone, head)
