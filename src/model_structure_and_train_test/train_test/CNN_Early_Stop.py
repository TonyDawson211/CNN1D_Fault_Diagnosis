import torch


def Early_Stop(ep, models, epoch_no_improve, epoch_patience, va_loss_avr, data_dir, best_epoch, best_va_loss_avr):
    # 设置最佳的验证损失率和最佳参数设定，初始化为0
    # 设置验证损失率未提升的轮次，忍耐轮次，初始化为0
    # 早停机制
    print("called")
    if va_loss_avr < best_va_loss_avr - 1e-5:
        best_va_loss_avr, best_epoch = va_loss_avr, ep  # 记录最佳损失率
        best_va_state = {k: v.cpu().clone() for k, v in models.state_dict().items()}
        torch.save(best_va_state, data_dir / "Model_Data" / "best_va_state.pth")
        epoch_no_improve = 0

    else:
        epoch_no_improve += 1  # 未打破最佳损失率，耐心计数一次

    if epoch_no_improve == epoch_patience:
        print("To avoid overfitting, trainning has stopped. \n"
              f"the data of chosen model: epoch{best_epoch}")

        return "break"

    print(best_va_loss_avr)
    return best_va_loss_avr
