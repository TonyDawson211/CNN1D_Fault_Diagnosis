import torch


# ================== CNN2D 主网络 ==================
class CNN2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fetures = torch.nn.Sequential(
            # ---- 第一个卷积块 ----
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # ---- 第二个卷积块 ----
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # ---- 全连接分类头 ----
            torch.nn.AdaptiveAvgPool2d((4, 30)),
            torch.nn.Flatten(),
            torch.nn.Linear(3840, 256),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.fetures(x)
