import torch
import numpy as np

from src.config.paths import DATA_DIR


def Trainning(rounds: int, devices, models, train_data, val_data, optimizers, criterions, pr=True):
    best_tr_acc_avr, best_tr_state = 0.0, None  # 设置最佳的验证准确率和最佳参数设定，初始化为0
    epoches = rounds  # 定义训练次数
    tr_loss_avr_list, tr_acc_avr_list, va_loss_avr_list, va_acc_avr_list = [], [], [], []  # 定义历史训练偏离程度，历史训练准确度，历史验证偏离程度，历史验证准确度列表，初始化为空列表
    for ep in range(1, epoches + 1):  # 训练迭代
        models.train()  # 选择训练模式
        tr_loss, tr_acc, n = 0.0, 0.0, 0  # 定义训练偏离程度，训练准确度，初始化为0
        for x_cpu, y_cpu in train_data:  # 迭代训练数据
            x, y = x_cpu.to(devices), y_cpu.to(devices)  # 将数据转移至GPU
            optimizers.zero_grad()  # 清空上一次迭代保留的梯度
            logits: torch.Tensor = models(x)  # 向前传播
            loss = criterions(logits, y)  # 计算损失
            loss.backward()  # 向后传播
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

        # 将每一次迭代的数据存储
        tr_loss_avr_list.append(tr_loss_avr)
        tr_acc_avr_list.append(tr_acc_avr)
        va_loss_avr_list.append(va_loss_avr)
        va_acc_avr_list.append(va_acc_avr)

        # 选择是否显示迭代数据
        if pr:
            print(f"epoch {ep:02d} | "
                  f"train loss {tr_loss_avr:.4f} | train acc {tr_acc_avr:.4f} | "
                  f"val loss {va_loss_avr:.4f} | val acc {va_acc_avr:.4f} | ")

        # 记录最佳验证精度和其对应的参数设置，并保存到文件
        if va_acc_avr > best_tr_acc_avr:
            best_tr_acc_avr = va_acc_avr
            best_tr_state = {k: v.cpu().clone() for k, v in models.state_dict().items()}
            torch.save(best_tr_state, DATA_DIR / "Model_Data/best_tr_state.pth")

    # 将迭代数据转换为array数组
    tr_loss_avr_arr = np.array(tr_loss_avr_list)
    tr_acc_avr_arr = np.array(tr_acc_avr_list)
    va_loss_avr_arr = np.array(va_loss_avr_list)
    va_acc_avr_arr = np.array(va_acc_avr_list)

    return best_tr_acc_avr, best_tr_state, tr_loss_avr_arr, tr_acc_avr_arr, va_loss_avr_arr, va_acc_avr_arr
