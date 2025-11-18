import torch.nn as nn

from src.enhancing_config.ChannelAttention1D import ChannelAttention1D
from src.enhancing_config.SpatialAttention1D import SpatialAttention1D


class CBAM1D(nn.Module):
    def __init__(self, in_channel, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention1D(in_channel, reduction)
        self.spatial_att = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = self.spatial_att(x)
        x = self.spatial_att(x)
        return x
