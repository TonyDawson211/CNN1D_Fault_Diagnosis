import glob
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pywt
from concurrent.futures import ProcessPoolExecutor


def Data_Cut_Col(chosen_col: list, stride, dtype: str, root):
    raw_data, raw_label = [], []

    for cls in os.listdir(root):  # 遍历数据根目录里面的每一个目录

        for file in glob.glob(os.path.join(root, cls, "*.csv")):  # 找到所有的csv文件
            data = pd.read_csv(file, header=None, dtype=dtype)  # 读取的每一个csv文件
            data = data.dropna(axis="columns", how="all")  # 去除空行
            data_col = data.iloc[:, chosen_col]  # 选取特定一列
            to_be_cut_data = data_col.to_numpy(dtype=dtype)  # 数据转换为float32

            for i in range(0, len(to_be_cut_data) - stride, stride):  # 以分割步长分割
                seg = (to_be_cut_data[i:i + stride] - np.mean(to_be_cut_data[i:i + stride])) / np.std(
                    to_be_cut_data[i:i + stride])  # 归一化
                raw_data.append(seg)  # 加入一组分割的数据
                raw_label.append(cls)  # 给其贴标签

    arr_data_out, arr_label_out = np.array(raw_data), np.array(raw_label)  # 将分割后的所有数据转换为array

    return arr_data_out, arr_label_out


def Encode_and_classify(data, label, test_ratio, val_ratio):
    le = LabelEncoder()  # 创建编码器
    label_int = le.fit_transform(label)  # 将原来的标签编序号，然后给数据贴上标签
    classes_out = le.classes_  # 类别编号的字典映射 # 0, 1, 2 -> 'inner', 'normal', 'outer'

    # 将分割后的总数据的0.2作为测试集
    data_trv, data_test_out, label_trv, label_test_out = train_test_split(data, label_int, test_size=test_ratio,
                                                                          random_state=42, stratify=label_int)
    # 总数据的0.15为验证集 -> 1 - 0.2 - 0.15为训练集
    data_train_out, data_val_out, label_train_out, label_val_out = train_test_split(data_trv, label_trv,
                                                                                    test_size=val_ratio / (
                                                                                            1 - test_ratio),
                                                                                    random_state=42, stratify=label_trv)

    return data_test_out, label_test_out, data_val_out, label_val_out, data_train_out, label_train_out, classes_out


def Set_Channel_Dim(matrix: np.ndarray):
    return np.moveaxis(matrix, source=-1, destination=1)


def Encode(input_data, input_label, method=None):
    (data_test, out_label_test, data_val,
     out_label_val, data_train, out_label_train, out_classes_) = Encode_and_classify(input_data, input_label, 0.2, 0.15)
    if method:
        data_test, data_train, data_val = method(data_test), method(
            data_train), method(
            data_val)

    out_data_test_c, out_data_train_c, out_data_val_c = (Set_Channel_Dim(data_test),
                                                         Set_Channel_Dim(data_train),
                                                         Set_Channel_Dim(data_val))

    return (out_classes_, out_data_test_c, out_data_train_c, out_data_val_c,
            out_label_test, out_label_val, out_label_train)


def one_cwt(single_batch: np.ndarray, num_scales: int = 64, wavelet: str = "morl"):
    scales = np.arange(1, num_scales + 1)
    coeffs, _ = pywt.cwt(single_batch, scales, wavelet)
    power = np.abs(coeffs) ** 2
    power_normalized = (power - power.mean()) / (power.std() + 1e-8)
    return power_normalized


def CWT_Scalogram_2D(matrix: np.ndarray):
    pool = [matrix[i] for i in range(matrix.shape[0])]
    cwt_signal = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for idx, res in enumerate(executor.map(one_cwt, pool)):
            cwt_signal.append(res)
            if idx % 10 == 0 or idx == len(pool):
                print(f"CWT 处理进度：{idx}/{len(pool)}")

    cwt_signal = np.array(cwt_signal)

    print(matrix.shape, cwt_signal.shape)
    return cwt_signal
