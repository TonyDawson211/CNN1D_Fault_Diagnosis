import numpy as np
import torch
from torch.utils.data import DataLoader
from src.enhancing_module.FusionVibDataset import FusionVibDataset
from src.config.paths import DATA_DIR
import os

# 读取存储的学习数据
data_test_c_1D = np.load(DATA_DIR / "data_test_1D.npy")
data_train_c_1D = np.load(DATA_DIR / "data_train_1D.npy")
data_val_c_1D = np.load(DATA_DIR / "data_val_1D.npy")

data_test_c_2D = np.load(DATA_DIR / "data_test_2D.npy")
data_train_c_2D = np.load(DATA_DIR / "data_train_2D.npy")
data_val_c_2D = np.load(DATA_DIR / "data_val_2D.npy")

label_test_1D = np.load(DATA_DIR / "label_test_1D.npy")
label_train_1D = np.load(DATA_DIR / "label_train_1D.npy")
label_val_1D = np.load(DATA_DIR / "label_val_1D.npy")

label_test_2D = np.load(DATA_DIR / "label_test_2D.npy")
label_train_2D = np.load(DATA_DIR / "label_train_2D.npy")
label_val_2D = np.load(DATA_DIR / "label_val_2D.npy")

num_worker = os.cpu_count()
# 构建PyTorch数据加载器
# 训练数据加载
train_ds = FusionVibDataset(torch.tensor(data_train_c_1D, dtype=torch.float32),
                            torch.tensor(data_train_c_2D, dtype=torch.float32),
                            torch.tensor(label_train_2D, dtype=torch.long),
                            is_augmented=True)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)  # type: ignore
# 验证数据加载
val_ds = FusionVibDataset(torch.tensor(data_val_c_1D, dtype=torch.float32),
                          torch.tensor(data_val_c_2D, dtype=torch.float32),
                          torch.tensor(label_val_2D, dtype=torch.long),
                          is_augmented=False)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)  # type: ignore
# 测试数据加载
test_ds = FusionVibDataset(torch.tensor(data_test_c_1D, dtype=torch.float32),
                           torch.tensor(data_test_c_2D, dtype=torch.float32),
                           torch.tensor(label_test_2D, dtype=torch.long),
                           is_augmented=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)  # type: ignore
