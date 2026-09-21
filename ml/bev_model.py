"""Compact U-Net style BEV segmenter for development training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.SiLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.SiLU(inplace=True),
        )


class BevSegmenter(nn.Module):
    def __init__(self, input_channels: int = 7, base_channels: int = 16):
        super().__init__()
        self.encoder1 = ConvBlock(input_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.decoder2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    def logits(self, value: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(value)
        second = self.encoder2(F.max_pool2d(first, 2))
        middle = self.bottleneck(F.max_pool2d(second, 2))
        up2 = F.interpolate(middle, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.decoder2(torch.cat((up2, second), dim=1))
        up1 = F.interpolate(up2, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.decoder1(torch.cat((up1, first), dim=1)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(value))
