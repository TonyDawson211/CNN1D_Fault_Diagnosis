import torch
from src.CNN_Model_Device import model, device
from CNN_Training_Data_Load import train_loader, val_loader
from CNN_Training import Trainning

# 选择优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# 选择损失类型
critirion = torch.nn.CrossEntropyLoss()

# 接受训练所得数据
best_tr_acc_avr, best_tr_state, tl, ta, vl, va = Trainning(int(input("输入迭代次数：")), device, model, train_loader, val_loader,
                                                           optimizer,
                                                           critirion)

model = model.load_state_dict(best_tr_state)  # 载入最佳训练参数模型


def Test(models, test_data, pr=True):
    models.eval()  # 选择验证模式来测试
    te_acc, te_n = 0.0, 0  # 定义测试正确额的样本数，测试的样本总数，初始化为0

    with torch.no_grad():  # 关闭梯度开始测试
        for x_cpu, y_cpu in test_data:
            x, y = x_cpu.to(device), y_cpu.to(device)
            logits: torch.Tensor = models(x)
            te_acc += (logits.argmax(1) == y).float().sum().item()
            te_n += y.size(0)
            te_acc_avr = te_acc / te_n  # 记录测试准确率

    if pr:
        print(f"ea {te_acc_avr} | b_ta {best_tr_acc_avr}")

    return te_acc_avr, best_tr_acc_avr, best_tr_state, tl, ta, vl, va
