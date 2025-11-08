import glob
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.paths import DATA_DIR, SIGNAL_DIR

data_dir = SIGNAL_DIR  # 定义数据根目录
stride = 2400  # 分割步长

raw_data, raw_label = [], []

for cls in os.listdir(data_dir):  # 遍历数据根目录里面的每一个目录
    for file in glob.glob(os.path.join(data_dir, cls, "*.csv")):  # 找到所有的csv文件
        data = pd.read_csv(file, header=None, dtype="float32")  # 读取的每一个csv文件
        data = data.dropna(axis="columns", how="all")  # 去除空行
        data_col_3 = data.iloc[:, 2]  # 选取特定一列
        to_be_cut_data = data_col_3.to_numpy(dtype="float32")  # 数据转换为float32
        for i in range(0, len(to_be_cut_data) - stride, stride):  # 以分割步长分割
            seg = (to_be_cut_data[i:i + stride] - np.mean(to_be_cut_data[i:i + stride])) / np.std(
                to_be_cut_data[i:i + stride])  # 归一化
            raw_data.append(seg)  # 加入一组分割的数据
            raw_label.append(cls)  # 给其贴标签
arr_data, arr_label = np.array(raw_data), np.array(raw_label)  # 将分割后的所有数据转换为array
# 保存为二进制数据格式
np.save(DATA_DIR / "Pre_Training_Data" / "arr_data.npy", arr_data)
np.save(DATA_DIR / "Pre_Training_Data" / "arr_label.npy", arr_label)

input_data, input_label = np.load(DATA_DIR / "Pre_Training_Data" / "arr_data.npy"), np.load(
    DATA_DIR / "Pre_Training_Data" / "arr_label.npy")  # 读取分割的数据
le = LabelEncoder()  # 创建编码器
input_label_int = le.fit_transform(input_label)  # 将原来的标签编序号，然后给数据贴上标签
num_classes = len(le.classes_)  # 记录有多少类别数 # 0, 1, 2 -> 'inner', 'normal', 'outer'
np.save(DATA_DIR / "Pre_Training_Data" / "label_classes.npy", le.classes_)  # 保存类别编号的字典映射

# 将分割后的总数据的0.2作为测试集
data_trv, data_test, label_trv, label_test = train_test_split(input_data, input_label_int, test_size=0.20,
                                                              random_state=42, stratify=input_label_int)
# 总数据的0.15为验证集 -> 1 - 0。2 - 0.15为训练集
data_train, data_val, label_train, label_val = train_test_split(data_trv, label_trv, test_size=0.15 / (1 - 0.20),
                                                                random_state=42, stratify=label_trv)


# 增加通道维 -> 1D-CNN
def add_channel_dim(martix: np.ndarray):
    return martix.reshape(martix.shape[0], 1, martix.shape[1])


data_test_c, data_train_c, data_val_c = add_channel_dim(data_test), add_channel_dim(data_train), add_channel_dim(
    data_val)

# 存储所有学习数据
np.save(DATA_DIR / "data_test.npy", data_test_c)
np.save(DATA_DIR / "data_train.npy", data_train_c)
np.save(DATA_DIR / "data_val.npy", data_val_c)
np.save(DATA_DIR / "label_test.npy", label_test)
np.save(DATA_DIR / "label_train.npy", label_train)
np.save(DATA_DIR / "label_val.npy", label_val)
