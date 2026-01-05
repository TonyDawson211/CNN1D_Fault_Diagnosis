import torch

from src.model_structure_and_train_test.model_structure.Freqency_Attention import Frequency_Attention_Module


# ================== CNN2D 主网络 ==================
class CNN2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            # ---- 第一个卷积块 ----
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # ---- 频带注意力模块 ----
            Frequency_Attention_Module(16),

            # ---- 第二个卷积块 ----
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # ---- 第三个卷积块 ----
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # ---- 全连接分类头 ----
            torch.nn.AdaptiveAvgPool2d((4, 12)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 12, 256),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)
