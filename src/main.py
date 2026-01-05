import torch
from src.model_structure_and_train_test.model_structure.CNN_Fusion import model, device
from src.model_structure_and_train_test.train_test.CNN_Test import Test
from src.model_structure_and_train_test.train_test.CNN_Train import Train
from src.data_prerequisite.CNN_Training_Data_Load import test_loader, train_loader, val_loader

# 选择优化器
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-3
)

# 选择损失类型
criterion = torch.nn.CrossEntropyLoss(
    label_smoothing=0.05  # 标签平滑，强制降低5%的正确率，降低过度自信
)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")  # 确保这里打印的是 cuda，而不是 cpu
    best_va_acc_avr, best_va_loss_avr, tl, ta, vl, va = Train(int(input("输入迭代次数：")),
                                                              device, model, train_loader,
                                                              val_loader,
                                                              optimizer,
                                                              criterion)
    te_acc_avr = Test(model, test_loader)
