import torch


def val_model(models, devices, criterions, val_data):
    models.eval()  # 选择验证模式
    va_loss, va_acc, number_batch_val = 0.0, 0.0, 0  # 定义验证偏离程度，验证准确度，初始化为0

    with torch.no_grad():  # 训练模式关闭，禁用梯度，生显存，提速
        for x_1d_cpu, x_2d_cpu, y_cpu in val_data:  # 迭代训练验证
            x_1d, x_2d, y = x_1d_cpu.to(devices), x_2d_cpu.to(devices), y_cpu.to(devices)  # 将数据转移至GPU
            logits = models(x_1d, x_2d)  # 向前传播
            loss = criterions(logits, y)  # 计算损失
            batch_size = y.size(0)  # 本次验证的样本数
            va_loss += loss.item() * batch_size  # 用样本数加权累计损失
            va_acc += (logits.argmax(1) == y).float().sum().item()  # type: ignore # 计算正确样本数
            number_batch_val += batch_size  # 累计已见样本数

    return va_loss, va_acc, number_batch_val
