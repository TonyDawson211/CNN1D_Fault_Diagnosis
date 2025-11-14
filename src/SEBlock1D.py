import torch


class SEBlock1D(torch.nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avr_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // reduction, channels, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        dim1, dim2, _ = x.size()

        y: torch.Tensor = self.avr_pool(x)
        y = y.view(dim1, dim2)
        y = self.fc(y)
        y = y.view(dim1, dim2, 1)

        return x * y.expand_as(x)
