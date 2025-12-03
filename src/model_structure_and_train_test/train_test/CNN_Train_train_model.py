import torch


def train_model(models, devices, criterions, optimizers, train_data):
    models.train()  # 选择训练模式
    tr_loss, tr_acc, number_batch_tr = 0.0, 0.0, 0  # 定义训练偏离程度，训练准确度，初始化为0

    for x_1d_cpu, x_2d_cpu, y_cpu in train_data:  # 迭代训练数据
        x_1d, x_2d, y = x_1d_cpu.to(devices), x_2d_cpu.to(devices), y_cpu.to(devices)  # 将数据转移至GPU
        optimizers.zero_grad()  # 清空上一次迭代保留的梯度
        logits: torch.Tensor = models(x_1d, x_2d)  # 向前传播
        loss = criterions(logits, y)  # 计算损失
        loss.backward()  # 向后传播
        torch.nn.utils.clip_grad_norm_(models.parameters(), max_norm=5.0)  # 梯度裁剪
        optimizers.step()  # 用梯度更新参数
        batch_size = y.size(0)  # 本次训练的样本数
        tr_loss += loss.item() * batch_size  # 用样本数加权累计损失
        tr_acc += (logits.argmax(1) == y).float().sum().item()  # 计算正确样本数 # type: ignore
        number_batch_tr += batch_size  # 累计已见样本数`

    return tr_loss, tr_acc, number_batch_tr
