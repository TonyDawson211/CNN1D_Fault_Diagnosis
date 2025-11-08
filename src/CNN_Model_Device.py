import torch
import numpy as np

from src.config.paths import DATA_DIR

# 加载标签的种类数
num_classes = int(len(np.load(DATA_DIR / "Pre_Training_Data/label_classes.npy")))


# 构建CNN1D神经网络
class CNN1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=20, stride=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            torch.nn.Conv1d(32, 64, kernel_size=10, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            torch.nn.Flatten(),
            torch.nn.Linear(64 * 57, 500),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(500, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# 激活神经网络模型
model = CNN1D()
# 选择运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
