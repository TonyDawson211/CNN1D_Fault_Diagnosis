import numpy as np
from src.config.paths import DATA_DIR, SIGNAL_DIR
from src.data_prerequisite.Prerequisite_Function import Data_Cut_Col


def Cut():
    out_arr_data, out_arr_label = Data_Cut_Col([2], 2400, "float32", SIGNAL_DIR)
    return out_arr_data, out_arr_label


if __name__ == "__main__":
    arr_data, arr_label = Cut()

    # 保存为二进制数据格式
    np.save(DATA_DIR / "Pre_Training_Data" / "arr_data.npy", arr_data)
    np.save(DATA_DIR / "Pre_Training_Data" / "arr_label.npy", arr_label)
