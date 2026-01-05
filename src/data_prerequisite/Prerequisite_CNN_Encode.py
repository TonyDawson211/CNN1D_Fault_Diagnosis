import numpy as np
from src.config.paths import DATA_DIR
from src.data_prerequisite.Prerequisite_Function import Encode, CWT_Scalogram_2D

input_data, input_label = np.load(DATA_DIR / "Pre_Training_Data" / "arr_data.npy"), np.load(
    DATA_DIR / "Pre_Training_Data" / "arr_label.npy")  # 读取分割的数据

if __name__ == "__main__":
    (classes_1D, data_test_c_1D, data_train_c_1D, data_val_c_1D,
     label_test_1D, label_val_1D, label_train_1D) = Encode(input_data, input_label)
    (classes_2D, data_test_c_2D, data_train_c_2D, data_val_c_2D,
     label_test_2D, label_val_2D, label_train_2D) = Encode(input_data, input_label, CWT_Scalogram_2D)

    # 存储所有学习数据
    np.save(DATA_DIR / "Pre_Training_Data" / "label_classes_1D.npy", classes_1D)  # 保存类别编号的字典映射
    np.save(DATA_DIR / "Pre_Training_Data" / "label_classes_2D.npy", classes_2D)

    np.save(DATA_DIR / "data_test_1D.npy", data_test_c_1D)
    np.save(DATA_DIR / "data_train_1D.npy", data_train_c_1D)
    np.save(DATA_DIR / "data_val_1D.npy", data_val_c_1D)

    np.save(DATA_DIR / "data_test_2D.npy", data_test_c_2D)
    np.save(DATA_DIR / "data_train_2D.npy", data_train_c_2D)
    np.save(DATA_DIR / "data_val_2D.npy", data_val_c_2D)

    np.save(DATA_DIR / "label_test_1D.npy", label_test_1D)
    np.save(DATA_DIR / "label_train_1D.npy", label_train_1D)
    np.save(DATA_DIR / "label_val_1D.npy", label_val_1D)

    np.save(DATA_DIR / "label_test_2D.npy", label_test_2D)
    np.save(DATA_DIR / "label_train_2D.npy", label_train_2D)
    np.save(DATA_DIR / "label_val_2D.npy", label_val_2D)
