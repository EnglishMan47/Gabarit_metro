# Adapted from BelousovArts/fod, commit baebf57a4169697aa61d017fcea8e2374ef135a0.
# No upstream license was published at that revision; see NOTICE.md.
"""Residual block copied from fod/rail_seg_net.py."""

import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.skip = nn.Conv2d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()
        self.a = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.1))
        self.b = nn.Sequential(nn.Conv2d(c_out, c_out, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c_out),
                               nn.LeakyReLU(0.1))
        self.c = nn.Sequential(nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.leaky_relu(self.c(self.b(self.a(x))) + self.skip(x), 0.1)

