import numpy as np
from src.config.paths import DATA_DIR
from src.data_prerequisite.Prerequisite_Function import Encoding, CWT_Scalogram_2D, Set_Channel_Dim

input_data, input_label = np.load(DATA_DIR / "Pre_Training_Data" / "arr_data.npy"), np.load(
    DATA_DIR / "Pre_Training_Data" / "arr_label.npy")  # 读取分割的数据


def Encode():
    (data_test, out_label_test, data_val,
     out_label_val, data_train, out_label_train, out_classes_) = Encoding(input_data,
                                                                          input_label,
                                                                          0.2,
                                                                          0.15)

    data_test, data_train, data_val = CWT_Scalogram_2D(data_test), CWT_Scalogram_2D(
        data_train), CWT_Scalogram_2D(
        data_val)

    out_data_test_c, out_data_train_c, out_data_val_c = (Set_Channel_Dim(data_test),
                                                         Set_Channel_Dim(data_train),
                                                         Set_Channel_Dim(data_val))

    return (out_classes_, out_data_test_c, out_data_train_c, out_data_val_c,
            out_label_test, out_label_val, out_label_train)


if __name__ == "__main__":
    classes_, data_test_c, data_train_c, data_val_c, label_test, label_val, label_train = Encode()

    # 存储所有学习数据
    np.save(DATA_DIR / "Pre_Training_Data" / "label_classes.npy", classes_)  # 保存类别编号的字典映射
    np.save(DATA_DIR / "data_test.npy", data_test_c)
    np.save(DATA_DIR / "data_train.npy", data_train_c)
    np.save(DATA_DIR / "data_val.npy", data_val_c)
    np.save(DATA_DIR / "label_test.npy", label_test)
    np.save(DATA_DIR / "label_train.npy", label_train)
    np.save(DATA_DIR / "label_val.npy", label_val)
