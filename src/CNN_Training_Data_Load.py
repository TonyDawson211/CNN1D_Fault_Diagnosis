import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.config.paths import DATA_DIR

# 读取存储的学习数据
data_test_c = np.load(DATA_DIR / "data_test.npy")
data_train_c = np.load(DATA_DIR / "data_train.npy")
data_val_c = np.load(DATA_DIR / "data_val.npy")
label_test = np.load(DATA_DIR / "label_test.npy")
label_train = np.load(DATA_DIR / "label_train.npy")
label_val = np.load(DATA_DIR / "label_val.npy")

# 构建PyTorch数据加载器
# 训练数据加载
train_ds = TensorDataset(torch.tensor(data_train_c, dtype=torch.float32), torch.tensor(label_train, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
# 验证数据加载
val_ds = TensorDataset(torch.tensor(data_val_c, dtype=torch.float32), torch.tensor(label_val, dtype=torch.long))
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
# 测试数据加载
test_ds = TensorDataset(torch.tensor(data_test_c, dtype=torch.float32), torch.tensor(label_test, dtype=torch.long))
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
