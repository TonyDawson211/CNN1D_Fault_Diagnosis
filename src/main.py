from src.CNN_Test import Test
from src.CNN_Model_Device import model
from src.CNN_Training_Data_Load import test_loader

if __name__ == "__main__":
    te_acc_avr, best_tr_acc_avr, best_va_state, tl, ta, vl, va = Test(model, test_loader)
