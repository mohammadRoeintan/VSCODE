import argparse
import pickle
import time
import os
from utils import Data, split_validation
# model.py شامل SessionGraph, train_test است. trans_to_cuda دیگر مستقیماً از اینجا ایمپورت نمی‌شود
# چون انتقال به دستگاه در خود model.py یا با .to(device) انجام می‌شود.
from model import SessionGraph, train_test
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')

# --- پارامترهای SSL ---
parser.add_argument('--ssl_weight', type=float, default=0.1, help='Weight for Self-Supervised Learning Loss')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='Temperature parameter for InfoNCE Loss')
parser.add_argument('--ssl_dropout_rate', type=float, default=0.2, help='Dropout rate for SSL augmentation')

# --- پارامترهای انکودر ترانسفورمر ---
parser.add_argument('--nhead', type=int, default=2, help='number of heads in transformer encoder')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers in transformer encoder')
parser.add_argument('--ff_hidden', type=int, default=256, help='dimension of feedforward network model in transformer')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in transformer')

opt = parser.parse_args()
print(opt)

# تعیین دستگاه به صورت گلوبال‌تر یا پاس دادن آن به توابع
# این device در model.py (مثلاً در تابع forward سراسری) استفاده خواهد شد.
# اگر مدل و داده‌ها مستقیماً روی این device ایجاد شوند، نیاز به trans_to_cuda کمتر می‌شود.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    base_dataset_path = './datasets/'

    if opt.dataset == 'sample':
        train_data_file = os.path.join(base_dataset_path, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, 'test.txt')
    else:
        train_data_file = os.path.join(base_dataset_path, opt.dataset, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, opt.dataset, 'test.txt')

    print(f"Loading training data from: {train_data_file}")
    try:
        # with open برای مدیریت بهتر فایل‌ها
        with open(train_data_file, 'rb') as f:
            train_data_loaded = pickle.load(f)
        if not (isinstance(train_data_loaded, tuple) and len(train_data_loaded) == 2):
            print(f"Error: Training data at {train_data_file} is not in the expected format (sessions, targets).")
            return
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_file}")
        return
    except Exception as e:
        print(f"Error loading or processing training data: {e}")
        return

    test_data_obj = None # مقدار اولیه
    if opt.validation:
        # اطمینان از اینکه train_data_loaded دارای داده است قبل از تقسیم
        if not train_data_loaded[0] or not train_data_loaded[1]:
            print("Error: Training data is empty, cannot perform validation split.")
            return
        train_set, valid_set = split_validation(train_data_loaded, opt.valid_portion)
        # test_data_loaded برای سازگاری با بخش n_node استفاده می‌شود، اگرچه test_data_obj از valid_set ساخته می‌شود
        test_data_loaded = valid_set
        try:
            train_data_obj = Data(train_set, shuffle=True)
            # در حالت validation، test_data_obj همان valid_set است
            test_data_obj = Data(valid_set, shuffle=False)
            print("Training and Validation Data objects created successfully.")
        except ValueError as e:
            print(f"Error creating Data objects for validation: {e}")
            return
    else:
        print(f"Loading testing data from: {test_data_file}")
        try:
            with open(test_data_file, 'rb') as f:
                test_data_loaded = pickle.load(f)
            if not (isinstance(test_data_loaded, tuple) and len(test_data_loaded) == 2):
                print(f"Error: Test data at {test_data_file} is not in the expected format (sessions, targets).")
                return
        except FileNotFoundError:
            print(f"Error: Testing data file not found at {test_data_file}. Proceeding without test data for n_node calculation if necessary.")
            test_data_loaded = ([], []) # اجازه ادامه برای محاسبه n_node فقط با داده ترین
        except Exception as e:
            print(f"Error loading or processing testing data: {e}")
            return
        try:
            train_data_obj = Data(train_data_loaded, shuffle=True)
            print("Training Data object created successfully.")
            if test_data_loaded[0] and test_data_loaded[1]: # فقط اگر داده تست معتبر بارگذاری شده باشد
                 test_data_obj = Data(test_data_loaded, shuffle=False)
                 print("Testing Data object created successfully.")
            else:
                 print("Warning: Test data is empty or was not found. Evaluation will not be performed unless in validation mode.")
        except ValueError as e:
            print(f"Error creating Data objects: {e}")
            return

    # محاسبه n_node
    # این بخش تغییر زیادی نکرده چون سربار آن یکبار در ابتدا است و معمولاً گلوگاه نیست.
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        all_nodes = set()
        if train_data_loaded and train_data_loaded[0]: # بررسی اینکه لیست جلسات خالی نباشد
            for session in train_data_loaded[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in train_data_loaded[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)

        if test_data_loaded and test_data_loaded[0]: # بررسی اینکه لیست جلسات تست خالی نباشد
            for session in test_data_loaded[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in test_data_loaded[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)

        if not all_nodes:
             print("Critical Warning: No nodes found in the dataset. Cannot determine n_node.")
             # بهتر است در این حالت برنامه متوقف شود یا n_node از طریق پارامتر ورودی گرفته شود
             # n_node = 310 if opt.dataset == 'sample' else 1000 # این مقدار پیش‌فرض ریسک بالایی دارد
             return # یا raise ValueError("No nodes found to determine n_node")
        else:
             n_node = max(all_nodes) + 1
        print(f"Calculated n_node based on actual data: {n_node}")

    if n_node <= 1: # پدینگ 0 و حداقل یک آیتم
        print(f"Error: Invalid n_node calculated: {n_node}. Must be > 1.")
        return

    model = SessionGraph(opt, n_node)
    model.to(device) # انتقال مدل به دستگاه (GPU یا CPU)

    checkpoint_dir = f'./checkpoints/{opt.dataset}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    start = time.time()
    best_result = [0, 0, 0]  # [recall, mrr, precision]
    best_epoch = [0, 0, 0]   # [epoch_for_best_recall, epoch_for_best_mrr, epoch_for_best_precision]
    bad_counter = 0

    # بررسی اینکه آیا داده تست/اعتبارسنجی برای ارزیابی وجود دارد
    if test_data_obj is None or test_data_obj.length == 0:
        print("Warning: No test/validation data available. Skipping evaluation loop and early stopping.")
        # اگر فقط آموزش مد نظر است، می‌توان حلقه را فقط برای آموزش اجرا کرد
        # یا برنامه را متوقف کرد. در اینجا فقط هشدار می‌دهیم.
        # برای سادگی، اگر داده تست نباشد، از ادامه حلقه اصلی جلوگیری می‌کنیم.
        # اگر می‌خواهید آموزش بدون ارزیابی انجام شود، این بخش را باید تغییر دهید.
        if not opt.validation: # اگر حالت اعتبارسنجی هم نبود، واقعا داده تست نداریم
            print("Exiting because no evaluation data is present and not in validation mode.")
            return


    for epoch_num in range(opt.epoch):
        print(f"{'-'*25} Epoch: {epoch_num} {'-'*25}")
        
        # اطمینان از اینکه train_data_obj و test_data_obj معتبر هستند
        if train_data_obj is None or train_data_obj.length == 0:
            print(f"Epoch {epoch_num}: No training data. Skipping epoch.")
            continue
        if test_data_obj is None or test_data_obj.length == 0:
             # این حالت نباید رخ دهد اگر بررسی اولیه انجام شده باشد
            print(f"Epoch {epoch_num}: No test/validation data. Skipping evaluation for this epoch.")
            # در این حالت، فقط آموزش می‌دهیم (اگر منطق train_test این را پشتیبانی کند)
            # یا اگر ارزیابی الزامی است، باید خطا داد یا از اپوک صرف نظر کرد.
            # فعلا فرض می‌کنیم train_test می‌تواند بدون داده تست اجرا شود و فقط آموزش دهد (باید در model.py بررسی شود)
            # اما برای جلوگیری از خطا، از فراخوانی train_test صرف نظر می‌کنیم اگر test_data_obj نباشد.
            # TODO: منطق train_test را بررسی کنید که آیا می‌تواند فقط آموزش دهد یا خیر.
            # برای سادگی فعلی، اگر test_data_obj نیست، از اپوک رد می‌شویم.
            continue


        # پاس دادن device به train_test می‌تواند مفید باشد اگر نیاز به ایجاد تنسور جدید روی دستگاه خاصی باشد
        # اما در پیاده‌سازی فعلی model.py، دستگاه از خود مدل یا ورودی‌ها گرفته می‌شود.
        recall, mrr, precision = train_test(model, train_data_obj, test_data_obj, opt)

        flag = 0
        if recall >= best_result[0]:
            best_result[0] = recall
            best_epoch[0] = epoch_num
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch_num
            flag = 1
        if precision >= best_result[2]:
            best_result[2] = precision
            best_epoch[2] = epoch_num
            flag = 1

        print('Current Best Result:')
        print(f'\tRecall@20: {best_result[0]:.4f}\tMRR@20: {best_result[1]:.4f}\tPrecision@20: {best_result[2]:.4f}\tEpochs: ({best_epoch[0]}, {best_epoch[1]}, {best_epoch[2]})')

        model_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch_num}.pth')
        # ذخیره state_dict مدل که شامل پارامترها است
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(), # برای امکان ادامه آموزش
            'scheduler_state_dict': model.scheduler.state_dict(), # برای امکان ادامه آموزش
            'best_result': best_result, # ذخیره نتایج برای اطلاع
            'opt': opt # ذخیره پارامترهای استفاده شده
        }, model_save_path)
        print(f'Model checkpoint saved to {model_save_path}')

        bad_counter += (1 - flag)
        if bad_counter >= opt.patience:
            print(f"Early stopping triggered after {opt.patience} epochs without improvement.")
            break

    print(f"{'-'*25} Training Finished {'-'*25}")
    end = time.time()
    print(f"Total Run time: {end - start:.2f} s")
    print("Final Best Overall Result:")
    print(f'\tRecall@20: {best_result[0]:.4f}\tMRR@20: {best_result[1]:.4f}\tPrecision@20: {best_result[2]:.4f}\tAchieved at Epochs: ({best_epoch[0]}, {best_epoch[1]}, {best_epoch[2]})')


if __name__ == '__main__':
    # تنظیم torch برای عدم استفاده از حافظه پنهان CUDA در برخی موارد (می‌تواند به جلوگیری از خطاهای حافظه کمک کند)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # در صورت نیاز
    
    # برای اطمینان از تکرارپذیری (اگرچه shuffle در Data وجود دارد)
    # torch.manual_seed(0)
    # np.random.seed(0)
    # if torch.cuda.is_available():
    # torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    main()