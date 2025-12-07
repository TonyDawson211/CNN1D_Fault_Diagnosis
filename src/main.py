from src.model_structure_and_train_test.train_test.CNN_Test import Test
from src.model_structure_and_train_test.model_structure.CNN_Fusion import model
from src.data_prerequisite.CNN_Training_Data_Load import test_loader

if __name__ == "__main__":
    te_acc_avr, best_tr_acc_avr, best_va_state, tl, ta, vl, va = Test(model, test_loader)
