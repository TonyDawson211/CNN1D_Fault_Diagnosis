import torch
import torch.nn as nn
import torch.nn.functional as func


class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        mid_channels = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        dim1, dim2, length = x.size
        avg_pool = func.adaptive_avg_pool1d(x, 1).view(dim1, dim2)
        max_pool, _ = func.adaptive_max_pool1d(x, 1).view(dim1, dim2)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        channel_att = self.sigmoid(avg_out + max_out).view(dim1, dim2, 1)
        out = x * channel_att
        return out
