import torch
from torch import nn, Tensor
from torch.nn import functional as F

from layers import ConvModule


class CondNetHead(nn.Module):
    def __init__(self, in_channels, num_classes: int, head_channels: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight_num = head_channels * num_classes
        self.bias_num = num_classes

        self.conv = ConvModule(in_channels[-1], head_channels, 1)
        self.dropout = nn.Dropout2d(0.1)

        self.guidance_project = nn.Conv2d(head_channels, num_classes, 1)
        self.filter_project = nn.Conv2d(
            head_channels * num_classes,
            self.weight_num + self.bias_num,
            1,
            groups=num_classes,
        )

    def forward(self, features) -> Tensor:
        x = self.dropout(self.conv(features[-1]))
        b, c, h, w = x.shape
        guidance_mask = self.guidance_project(x)
        cond_logit = guidance_mask

        key = x
        value = x
        guidance_mask = guidance_mask.softmax(dim=1).view(*guidance_mask.shape[:2], -1)
        key = key.view(b, c, -1).permute(0, 2, 1)

        cond_filters = torch.matmul(guidance_mask, key)
        cond_filters /= h * w
        cond_filters = cond_filters.view(b, -1, 1, 1)
        cond_filters = self.filter_project(cond_filters)
        cond_filters = cond_filters.view(b, -1)

        weight, bias = torch.split(cond_filters, [self.weight_num, self.bias_num], dim=1)
        weight = weight.reshape(b * self.num_classes, -1, 1, 1)
        bias = bias.reshape(b * self.num_classes)

        value = value.view(-1, h, w).unsqueeze(0)
        seg_logit = F.conv2d(value, weight, bias, 1, 0, groups=b).view(b, self.num_classes, h, w)

        if self.training:
            return cond_logit, seg_logit
        return seg_logit
