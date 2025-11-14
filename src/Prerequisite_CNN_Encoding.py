import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.config.paths import DATA_DIR

input_data, input_label = np.load(DATA_DIR / "Pre_Training_Data" / "arr_data.npy"), np.load(
    DATA_DIR / "Pre_Training_Data" / "arr_label.npy")  # 读取分割的数据
le = LabelEncoder()  # 创建编码器
input_label_int = le.fit_transform(input_label)  # 将原来的标签编序号，然后给数据贴上标签
num_classes = len(le.classes_)  # 记录有多少类别数 # 0, 1, 2 -> 'inner', 'normal', 'outer'
np.save(DATA_DIR / "Pre_Training_Data" / "label_classes.npy", le.classes_)  # 保存类别编号的字典映射

# 将分割后的总数据的0.2作为测试集
data_trv, data_test, label_trv, label_test = train_test_split(input_data, input_label_int, test_size=0.20,
                                                              random_state=42, stratify=input_label_int)

# 总数据的0.15为验证集 -> 1 - 0.2 - 0.15为训练集
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
