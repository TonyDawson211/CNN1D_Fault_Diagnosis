import torch


def train_model(models, devices, criterions, optimizers, train_data):
    models.train()  # 选择训练模式
    tr_loss, tr_acc, number_batch_tr = 0.0, 0.0, 0

    # 1. 获取总批次数量 (用于显示进度，比如 5/100)
    total_batches = len(train_data)

    # 2. 使用 enumerate 获取当前跑到第几批了 (step)
    for step, (x_1d_cpu, x_2d_cpu, y_cpu) in enumerate(train_data):
        x_1d, x_2d, y = x_1d_cpu.to(devices), x_2d_cpu.to(devices), y_cpu.to(devices)

        optimizers.zero_grad()
        logits: torch.Tensor = models(x_1d, x_2d)
        loss = criterions(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), max_norm=5.0)
        optimizers.step()

        batch_size = y.size(0)
        tr_loss += loss.item() * batch_size
        tr_acc += (logits.argmax(1) == y).float().sum().item()
        number_batch_tr += batch_size

        # =======================================================
        # 【新增功能】每训练 10 个 Batch，就打印一次日志
        # =======================================================
        if (step + 1) % 10 == 0:
            current_acc = (logits.argmax(1) == y).float().mean().item()
            print(
                f"正在训练: [{step + 1}/{total_batches}] | 当前Batch Loss: {loss.item():.4f} | 当前Batch Acc: {current_acc:.4f}")

    return tr_loss, tr_acc, number_batch_tr
