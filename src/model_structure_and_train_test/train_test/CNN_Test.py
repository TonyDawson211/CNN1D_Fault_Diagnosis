import torch
from src.config.paths import DATA_DIR
from src.model_structure_and_train_test.model_structure.CNN_Fusion import model, device


def Test(models, test_data):
    best_va_state = torch.load(DATA_DIR / "Model_Data" / "best_va_state.pth")
    model.load_state_dict(best_va_state)  # 载入最佳训练参数模型
    models.eval()  # 选择验证模式来测试
    te_acc, te_n = 0.0, 0  # 定义测试正确额的样本数，测试的样本总数，初始化为0

    with torch.no_grad():  # 关闭梯度开始测试
        for x_1d_cpu, x_2d_cpu, y_cpu in test_data:
            x_1d, x_2d, y = x_1d_cpu.to(device), x_2d_cpu.to(device), y_cpu.to(device)
            logits: torch.Tensor = models(x_1d, x_2d)
            te_acc += (logits.argmax(1) == y).float().sum().item()
            te_n += y.size(0)
            te_acc_avr = te_acc / te_n  # 记录测试准确率

    print(f"ea {te_acc_avr}")

    return te_acc_avr
