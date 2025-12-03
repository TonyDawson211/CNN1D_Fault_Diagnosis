import torch


# ================== CNN1D 主网络 ==================
class CNN1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # ---- 第一个卷积块 ----
            torch.nn.Conv1d(1, 32, kernel_size=20, stride=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            # ---- 第二个卷积块 ----
            torch.nn.Conv1d(32, 64, kernel_size=10, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # ---- 全连接分类头 ----
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 57, 256),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ================== 激活模型 & 选择设备 ==================
model = CNN1D()
# 选择运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
