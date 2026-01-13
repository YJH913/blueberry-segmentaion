from typing import Optional

from torch import nn

from .convnext import ConvNeXt, TorchvisionConvNeXtTBackbone
from .mit import MiT, TimmMiTB0Backbone
from .mobilenetv2 import MobileNetV2, TorchvisionMobileNetV2Backbone
from .resnet import ResNet, TorchvisionResNet50Backbone


def build_backbone(name: str, pretrained: bool = False, weights_path: Optional[str] = None) -> nn.Module:
    key = name.lower()
    if key == "resnet-50":
        if pretrained or weights_path:
            return TorchvisionResNet50Backbone(pretrained, weights_path)
        return ResNet("50")
    if key == "resnet-18":
        return ResNet("18")
    if key == "resnet-34":
        return ResNet("34")
    if key == "resnet-101":
        return ResNet("101")
    if key == "resnet-152":
        return ResNet("152")
    if key.startswith("mobilenetv2"):
        if pretrained or weights_path:
            return TorchvisionMobileNetV2Backbone(pretrained, weights_path)
        return MobileNetV2()
    if key == "mit-b0":
        if pretrained or weights_path:
            return TimmMiTB0Backbone(pretrained, weights_path)
        return MiT("B0")
    if key == "mit-b1":
        return MiT("B1")
    if key == "mit-b2":
        return MiT("B2")
    if key == "mit-b3":
        return MiT("B3")
    if key == "mit-b4":
        return MiT("B4")
    if key == "mit-b5":
        return MiT("B5")
    if key == "convnext-t":
        if pretrained or weights_path:
            return TorchvisionConvNeXtTBackbone(pretrained, weights_path)
        return ConvNeXt("T")
    if key == "convnext-s":
        return ConvNeXt("S")
    if key == "convnext-b":
        return ConvNeXt("B")
    raise ValueError(f"지원하지 않는 BACKBONE입니다: {name}")
