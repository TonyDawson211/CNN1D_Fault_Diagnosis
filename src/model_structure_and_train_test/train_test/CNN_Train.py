import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 学习率调度器
from src.config.paths import DATA_DIR
from src.model_structure_and_train_test.train_test.CNN_Early_Stop import Early_Stop
from src.model_structure_and_train_test.train_test.CNN_Train_train_model import train_model
from src.model_structure_and_train_test.train_test.CNN_Train_val_model import val_model


def Training(rounds: int, devices, models, train_data, val_data, optimizers, criterions):
    epoches = rounds  # 定义训练次数
    scheduler = ReduceLROnPlateau(optimizers, mode="min", factor=0.5, patience=2, min_lr=1e-6)  # 设置学习率调度器

    best_va_acc_avr, best_va_loss_avr, best_epoch = 0.0, np.inf, 0  # 设置最佳的验证准确率，最佳验证损失，初始化为0.0

    # 定义历史训练偏离程度，历史训练准确度，历史验证偏离程度，历史验证准确度列表，初始化为空列表
    tr_loss_avr_list, tr_acc_avr_list, va_loss_avr_list, va_acc_avr_list = [], [], [], []

    for ep in range(1, epoches + 1):  # 训练迭代

        tr_loss, tr_acc, number_batch_tr = train_model(models, devices, criterions, optimizers, train_data)  # 训练模式
        va_loss, va_acc, number_batch_val = val_model(models, devices, criterions, val_data)  # 验证模式

        tr_loss_avr, tr_acc_avr = tr_loss / number_batch_tr, tr_acc / number_batch_tr  # 计算训练损失，训练精度
        va_loss_avr, va_acc_avr = va_loss / number_batch_val, va_acc / number_batch_val  # 计算验证损失，验证精度

        # 学习律调度
        pre_lr = optimizers.param_groups[0]['lr']
        scheduler.step(va_loss_avr)  # 当vl连续指定轮数下降，就触发学习率调度
        new_lr = optimizers.param_groups[0]['lr']

        # 将每一次迭代的数据存储
        tr_loss_avr_list.append(tr_loss_avr)
        tr_acc_avr_list.append(tr_acc_avr)
        va_loss_avr_list.append(va_loss_avr)
        va_acc_avr_list.append(va_acc_avr)

        # 显示迭代数据
        print(f"epoch {ep:02d} | "
              f"train loss {tr_loss_avr:.4f} | train acc {tr_acc_avr:.4f} | "
              f"val loss {va_loss_avr:.4f} | val acc {va_acc_avr:.4f} | "
              f"pre_lr {pre_lr:.2e} -> new_lr {new_lr:.2e}")

        # 记录最佳验证准确率
        if va_acc_avr >= best_va_acc_avr:
            best_va_acc_avr = va_acc_avr

        # 早停机制
        judge = Early_Stop(ep, models, epoch_no_improve=0, epoch_patience=6, va_loss_avr=va_loss_avr,
                           data_dir=DATA_DIR, best_epoch=best_epoch, best_va_loss_avr=best_va_loss_avr)
        if judge == "break":
            break
        best_va_loss_avr = judge

    # 将迭代数据转换为array数组
    tr_loss_avr_arr = np.array(tr_loss_avr_list)
    tr_acc_avr_arr = np.array(tr_acc_avr_list)
    va_loss_avr_arr = np.array(va_loss_avr_list)
    va_acc_avr_arr = np.array(va_acc_avr_list)

    return (best_va_acc_avr, best_va_loss_avr,
            tr_loss_avr_arr, tr_acc_avr_arr, va_loss_avr_arr, va_acc_avr_arr)
