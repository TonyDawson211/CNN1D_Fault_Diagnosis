import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 学习率调度器
from src.config.paths import DATA_DIR


def Trainning(rounds: int, devices, models, train_data, val_data, optimizers, criterions, training_patience):
    epoches = rounds  # 定义训练次数
    scheduler = ReduceLROnPlateau(optimizers, mode="min", factor=0.5, patience=2, min_lr=1e-6)  # 设置学习率调度器

    best_va_loss_avr, best_va_state, best_epoch = np.inf, None, 0  # 设置最佳的验证损失率和最佳参数设定，初始化为0
    epoch_no_improve, epoch_patience = 0, training_patience  # 设置验证损失率未提升的轮次，忍耐轮次，初始化为0
    best_tr_acc_avr = 0.0  # 设置最佳的验证准确率和最佳参数设定，初始化为0

    # 定义历史训练偏离程度，历史训练准确度，历史验证偏离程度，历史验证准确度列表，初始化为空列表
    tr_loss_avr_list, tr_acc_avr_list, va_loss_avr_list, va_acc_avr_list = [], [], [], []

    for ep in range(1, epoches + 1):  # 训练迭代

        models.train()  # 选择训练模式
        tr_loss, tr_acc, n = 0.0, 0.0, 0  # 定义训练偏离程度，训练准确度，初始化为0

        for x_cpu, y_cpu in train_data:  # 迭代训练数据
            x, y = x_cpu.to(devices), y_cpu.to(devices)  # 将数据转移至GPU
            optimizers.zero_grad()  # 清空上一次迭代保留的梯度
            logits: torch.Tensor = models(x)  # 向前传播
            loss = criterions(logits, y)  # 计算损失
            loss.backward()  # 向后传播
            torch.nn.utils.clip_grad_norm_(models.parameters(), max_norm=5.0)  # 梯度裁剪
            optimizers.step()  # 用梯度更新参数
            batch_size = y.size(0)  # 本次训练的样本数
            tr_loss += loss.item() * batch_size  # 用样本数加权累计损失
            tr_acc += (logits.argmax(1) == y).float().sum().item()  # 计算正确样本数 # type: ignore
            n += batch_size  # 累计已见样本数

        models.eval()  # 选择验证模式
        va_loss, va_acc, m = 0.0, 0.0, 0  # 定义验证偏离程度，验证准确度，初始化为0

        with torch.no_grad():  # 训练模式关闭，禁用梯度，生显存，提速
            for x_cpu, y_cpu in val_data:  # 迭代训练验证
                x, y = x_cpu.to(devices), y_cpu.to(devices)  # 将数据转移至GPU
                logits = models(x)  # 向前传播
                loss = criterions(logits, y)  # 计算损失
                batch_size = y.size(0)  # 本次验证的样本数
                va_loss += loss.item() * batch_size  # 用样本数加权累计损失
                va_acc += (logits.argmax(1) == y).float().sum().item()  # 计算正确样本数
                m += batch_size  # 累计已见样本数

        tr_loss_avr, tr_acc_avr = tr_loss / n, tr_acc / n  # 计算训练损失，训练精度
        va_loss_avr, va_acc_avr = va_loss / m, va_acc / m  # 计算验证损失，验证精度

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
        if va_acc_avr >= best_tr_acc_avr:
            best_tr_acc_avr = va_acc_avr

        # 早停机制
        if va_loss_avr < best_va_loss_avr - 1e-5:
            best_va_loss_avr, best_epoch = va_loss_avr, ep  # 记录最佳损失率
            best_va_state = {k: v.cpu().clone() for k, v in models.state_dict().items()}
            torch.save(best_va_state, DATA_DIR / "Model_Data" / "best_va_state.pth")
            epoch_no_improve = 0

        else:
            epoch_no_improve += 1  # 未打破最佳损失率，耐心计数一次

        if epoch_no_improve == epoch_patience:
            print("To avoid overfitting, trainning has stopped. \n"
                  f"the data of chosen model: epoch{best_epoch}")
            break

    # 将迭代数据转换为array数组
    tr_loss_avr_arr = np.array(tr_loss_avr_list)
    tr_acc_avr_arr = np.array(tr_acc_avr_list)
    va_loss_avr_arr = np.array(va_loss_avr_list)
    va_acc_avr_arr = np.array(va_acc_avr_list)

    return (best_tr_acc_avr, best_va_loss_avr, best_va_state,
            tr_loss_avr_arr, tr_acc_avr_arr, va_loss_avr_arr, va_acc_avr_arr)
