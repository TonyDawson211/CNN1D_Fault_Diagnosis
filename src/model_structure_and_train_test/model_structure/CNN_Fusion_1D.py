import torch


# ================== CNN1D 主网络 ==================
class CNN1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            # ---- 第一个卷积块 ----
            torch.nn.Conv1d(1, 32, kernel_size=20, stride=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            # ---- 第二个卷积块 ----
            torch.nn.Conv1d(32, 64, kernel_size=10, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            # ---- 全连接分类头 ----
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 57, 256),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        return x
