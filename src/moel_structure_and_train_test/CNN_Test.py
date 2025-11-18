import torch
from src.moel_structure_and_train_test.CNN_Model_Device import model, device
from src.data_prerequisite.CNN_Training_Data_Load import train_loader, val_loader
from src.moel_structure_and_train_test.CNN_Train import Trainning

# 选择优化器
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-3
)

# 选择损失类型
critirion = torch.nn.CrossEntropyLoss(
    label_smoothing=0.05  # 标签平滑，强制降低5%的正确率，降低过度自信
)

# 接受训练所得数据
best_tr_acc_avr, best_va_loss_avr, best_va_state, tl, ta, vl, va = Trainning(int(input("输入迭代次数：")),
                                                                             device, model, train_loader,
                                                                             val_loader,
                                                                             optimizer,
                                                                             critirion, training_patience=6)

model.load_state_dict(best_va_state)  # 载入最佳训练参数模型
model.eval()


def Test(models, test_data):
    models.eval()  # 选择验证模式来测试
    te_acc, te_n = 0.0, 0  # 定义测试正确额的样本数，测试的样本总数，初始化为0

    with torch.no_grad():  # 关闭梯度开始测试
        for x_cpu, y_cpu in test_data:
            x, y = x_cpu.to(device), y_cpu.to(device)
            logits: torch.Tensor = models(x)
            te_acc += (logits.argmax(1) == y).float().sum().item()
            te_n += y.size(0)
            te_acc_avr = te_acc / te_n  # 记录测试准确率

    print(f"ea {te_acc_avr} | b_ta {best_tr_acc_avr} | b_vl {best_va_loss_avr}")

    return te_acc_avr, best_tr_acc_avr, best_va_state, tl, ta, vl, va
