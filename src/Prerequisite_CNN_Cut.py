import glob
import numpy as np
import os
import pandas as pd
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