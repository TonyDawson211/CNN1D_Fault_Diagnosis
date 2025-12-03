# CNN Bearing Fault Diagnosis

## 一维 CNN 复现滚动轴承故障诊断（基于原论文思路 + 自制数据处理流程）

## Attribution
- 基于：Rolling Element Bearings Fault Intelligent Diagnosis Based on Convolutional Neural Networks Using Raw Sensing Signal
- 代码参考：https://github.com/ZhangWei1993/Mechanical-Fault-Diagnosis-Based-on-Deep-Learning

## 环境
- Python 3.12.12
- 依赖：`pip install -r requirements.txt`

## 数据
- 大体量数据不随仓库提供。请把数据放到 `ROOT/Signal_Data/`下，请将数据对应标签放置好，一类故障的数据放在指定文件夹
- 结构自定；代码默认使用相对路径，详见 `ROOT/src/config`

## 代码介绍
- Signal_Data放入故障数据，里面有四种故障分类
- Training_Data存储了训练-验证-测试过程中的所有数据
  - 此目录下直接放置训练-验证-测试数据集
  - Model_Data下放置了训练过程中验证准确率最高的一次迭代的参数配置
  - Pre_Training_Data下放置了训练前预处理的数据
- src是核心代码区
  - data_prerequisite
    - Prerequisite_CNN_Cut, Prerequisite_CNN_Encoding 数据预处理
    - Prerequisite_Function 数据预处理函数
    - CNN_Training_Data_load 数据转化为训练可用数据
  - enhancing_config 强化模块配置
    - TimeSeriesAugment 数据增强
  - enhancing_module 强化模块
    - FusionVibDataset 强化数据集
  - model_structure_and_train_test 模块搭建与训练测试
    - model structure 神经网络框架和设备搭建
    - train test 数据的训练和测试
      - CNN_Train 训练框架搭建
      - CNN_Test 测试框架搭建
      - 其余的为组装模块
  - config 存储的运行的路径配置
  - src_jupyter 核心代码的jupyter版本，用于调试
  - main 总运行接口
## 开始
- 先运行 Prerequisite_CNN_Cut和Prerequisite_CNN_Encoding作数据预处理，再运行 main 总接口
