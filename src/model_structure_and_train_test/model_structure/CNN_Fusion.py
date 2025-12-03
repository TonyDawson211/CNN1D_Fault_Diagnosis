import numpy as np
import torch

from src.config.paths import DATA_DIR
from src.model_structure_and_train_test.model_structure.CNN_Fusion_1D import CNN1D
from src.model_structure_and_train_test.model_structure.CNN_Fusion_2D import CNN2D

num_classes = int(len(np.load(DATA_DIR / "Pre_Training_Data/label_classes_1D.npy")))


# ================== CNN Fusion主网络 ==================
class FusionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_1d = CNN1D()
        self.branch_2d = CNN2D()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 + 256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x_1d, x_2d):
        fusion_1d = self.branch_1d(x_1d)
        fusion_2d = self.branch_2d(x_2d)
        fusion = torch.cat((fusion_1d, fusion_2d), dim=1)
        out = self.classifier(fusion)

        return out


# ================== 激活模型 & 选择设备 ==================
model = FusionNet()
# 选择运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
