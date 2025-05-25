import argparse
import pickle
import time
import os
# import random # اگر می‌خواهید نمونه‌برداری تصادفی انجام دهید، این را اضافه کنید
from utils import Data, split_validation
from model import SessionGraph, trans_to_cuda, train_test
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
parser.add_argument('--data_subset_ratio', type=float, default=1.0, help='Factor to reduce dataset size (e.g., 0.1 for 10%). Default is 1.0 (full data).') # آرگومان جدید

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    base_dataset_path = './datasets/'
    if opt.dataset == 'sample':
        train_data_file = base_dataset_path + 'train.txt'
        test_data_file = base_dataset_path + 'test.txt'
    else:
        train_data_file = base_dataset_path + opt.dataset + '/train.txt'
        test_data_file = base_dataset_path + opt.dataset + '/test.txt'

    print(f"Loading training data from: {train_data_file}")
    try:
        train_data_loaded = pickle.load(open(train_data_file, 'rb'))
        if not (isinstance(train_data_loaded, tuple) and len(train_data_loaded) == 2 and
                isinstance(train_data_loaded[0], list) and isinstance(train_data_loaded[1], list)):
            print(f"Error: Training data at {train_data_file} is not in the expected format (list_of_sessions, list_of_targets).")
            return
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_file}")
        return
    except Exception as e:
        print(f"Error loading or processing training data: {e}")
        return

    # --- محدود کردن داده آموزشی بر اساس data_subset_ratio ---
    if 0 < opt.data_subset_ratio < 1.0:
        train_sessions, train_targets = train_data_loaded
        num_original_train_samples = len(train_sessions)
        if num_original_train_samples > 0:
            num_train_samples_to_keep = int(num_original_train_samples * opt.data_subset_ratio)
            if num_train_samples_to_keep == 0: # حداقل یک نمونه نگه دار اگر داده اصلی وجود دارد
                num_train_samples_to_keep = 1
            
            # برای نمونه‌برداری نماینده‌تر، می‌توانید ابتدا داده‌ها را shuffle کنید:
            # combined_train = list(zip(train_sessions, train_targets))
            # random.shuffle(combined_train)
            # train_sessions_shuffled, train_targets_shuffled = zip(*combined_train)
            # train_sessions_limited = list(train_sessions_shuffled[:num_train_samples_to_keep])
            # train_targets_limited = list(train_targets_shuffled[:num_train_samples_to_keep])
            # train_data_loaded = (train_sessions_limited, train_targets_limited)
            
            # یا به سادگی N نمونه اول را بگیرید:
            train_data_loaded = (train_sessions[:num_train_samples_to_keep], train_targets[:num_train_samples_to_keep])
            print(f"--- Limiting training data to {len(train_data_loaded[0])} samples ({opt.data_subset_ratio*100:.1f}%) for testing. ---")
        else:
            print("--- Training data is empty, no subsetting possible. ---")
    # ---------------------------------------------------------

    train_set_for_data_obj = train_data_loaded # مقدار اولیه برای train_set
    test_data_for_data_obj = ([], [])      # مقدار اولیه برای test_set

    if opt.validation:
        print(f"Splitting training data for validation (portion: {opt.valid_portion})")
        # train_data_loaded در این مرحله ممکن است از قبل محدود شده باشد
        train_set_for_data_obj, valid_set_for_data_obj = split_validation(train_data_loaded, opt.valid_portion)
        test_data_for_data_obj = valid_set_for_data_obj # مجموعه تست همان مجموعه اعتبارسنجی است
        print(f"Number of training samples after split: {len(train_set_for_data_obj[0])}")
        print(f"Number of validation samples: {len(test_data_for_data_obj[0])}")
    else:
        print(f"Loading testing data from: {test_data_file}")
        try:
            test_data_loaded_full = pickle.load(open(test_data_file, 'rb'))
            if not (isinstance(test_data_loaded_full, tuple) and len(test_data_loaded_full) == 2 and
                    isinstance(test_data_loaded_full[0], list) and isinstance(test_data_loaded_full[1], list)):
                print(f"Error: Test data at {test_data_file} is not in the expected format (list_of_sessions, list_of_targets).")
                return # یا یک مقدار پیش‌فرض برای test_data_for_data_obj قرار دهید
            
            test_data_for_data_obj = test_data_loaded_full # مقدار اولیه

            # --- محدود کردن داده تست بر اساس data_subset_ratio ---
            if 0 < opt.data_subset_ratio < 1.0:
                test_sessions, test_targets = test_data_loaded_full
                num_original_test_samples = len(test_sessions)
                if num_original_test_samples > 0:
                    num_test_samples_to_keep = int(num_original_test_samples * opt.data_subset_ratio)
                    if num_test_samples_to_keep == 0: # حداقل یک نمونه
                        num_test_samples_to_keep = 1
                    
                    test_data_for_data_obj = (test_sessions[:num_test_samples_to_keep], test_targets[:num_test_samples_to_keep])
                    print(f"--- Limiting testing data to {len(test_data_for_data_obj[0])} samples ({opt.data_subset_ratio*100:.1f}%) for testing. ---")
                else:
                    print("--- Test data is empty, no subsetting possible. ---")
            # ---------------------------------------------------------

        except FileNotFoundError:
            print(f"Error: Testing data file not found at {test_data_file}. Using empty test set.")
            test_data_for_data_obj = ([], [])
        except Exception as e:
            print(f"Error loading or processing testing data: {e}. Using empty test set.")
            test_data_for_data_obj = ([], [])
        print(f"Number of testing samples: {len(test_data_for_data_obj[0])}")


    # محاسبه n_node بر اساس داده‌های واقعی (داده‌های آموزشی اصلی قبل از محدودسازی برای n_node بهتر است،
    # اما برای تست سریع، می‌توان از داده محدود شده هم استفاده کرد، به شرطی که همه آیتم‌ها را پوشش دهد)
    # برای اطمینان، n_node را بر اساس داده‌های کامل اولیه محاسبه می‌کنیم (اگر تغییر نکرده باشند)
    # یا می‌توانیم یک کپی از داده‌های کامل را برای محاسبه n_node نگه داریم.
    # در اینجا، n_node بر اساس داده‌های train_data_loaded (که ممکن است محدود شده باشد) و test_data_loaded_full (اگر validation=False) محاسبه می‌شود.
    # این می‌تواند باعث شود n_node کوچکتر از حد واقعی باشد اگر آیتم‌های با ID بالا در بخش محدود شده نباشند.
    # برای تست سریع، این ممکن است قابل قبول باشد.
    
    n_node_calculation_train_data = train_data_loaded # داده آموزشی (احتمالا محدود شده)
    n_node_calculation_test_data = test_data_for_data_obj if not opt.validation else ([],[]) # داده تست (احتمالا محدود شده) یا خالی

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        all_nodes = set()
        # استخراج نودها از داده آموزشی
        if n_node_calculation_train_data and n_node_calculation_train_data[0]:
            for session in n_node_calculation_train_data[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in n_node_calculation_train_data[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)

        # استخراج نودها از داده تست (اگر وجود داشته باشد و validation نباشد)
        if n_node_calculation_test_data and n_node_calculation_test_data[0]:
            for session in n_node_calculation_test_data[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in n_node_calculation_test_data[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)

        if not all_nodes:
             print("Warning: No nodes found in the (potentially subsetted) dataset. Setting n_node to a default for 'sample'. This might cause issues.")
             n_node = 310 if opt.dataset == 'sample' else 1000
        else:
             n_node = max(all_nodes) + 1
        print(f"Calculated n_node based on (potentially subsetted) data: {n_node}")

    if n_node <= 1:
        print(f"Error: Invalid n_node calculated: {n_node}. Check dataset processing or subset ratio.")
        return
        
    # ساخت آبجکت‌های Data با داده‌های نهایی (احتمالاً محدود شده)
    try:
        train_data_obj = Data(train_set_for_data_obj, shuffle=True)
        print(f"Training Data object created successfully with {len(train_data_obj.inputs)} samples.")
    except ValueError as e:
        print(f"Error creating Training Data object: {e}")
        return

    try:
        test_data_obj = Data(test_data_for_data_obj, shuffle=False)
        print(f"Testing Data object created successfully with {len(test_data_obj.inputs)} samples.")
    except ValueError as e:
        print(f"Error creating Testing Data object: {e}")
        return


    model = SessionGraph(opt, n_node)
    model = trans_to_cuda(model)

    checkpoint_dir = f'./checkpoints/{opt.dataset}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    start = time.time()
    best_result = [0, 0, 0]
    best_epoch = [0, 0, 0]
    bad_counter = 0

    for epoch_num in range(opt.epoch):
        print(f"{'-'*25} Epoch: {epoch_num} {'-'*25}")
        # اطمینان از اینکه train_data_obj و test_data_obj داده دارند
        if len(train_data_obj.inputs) == 0 :
            print("Skipping training for epoch as training data is empty.")
            if not opt.validation and len(test_data_obj.inputs) == 0:
                 print("Both training and testing data are empty. Stopping.")
                 break # یا continue اگر می‌خواهید فقط این اپوک رد شود
            # اگر فقط داده آموزشی خالی است، ممکن است بخواهید همچنان تست کنید (اگر داده تست وجود دارد)
            # یا خطا دهید. در اینجا فرض می‌کنیم آموزش ممکن نیست.
            recall, mrr, precision = 0,0,0 # یا مقادیر قبلی
        elif not opt.validation and len(test_data_obj.inputs) == 0 and epoch_num > 0: # اگر داده تست خالی است و اولین اپوک نیست
            print("Skipping testing for epoch as testing data is empty (and not in validation mode).")
            # فقط آموزش انجام می‌شود اگر بخواهید، اما train_test نیاز به داده تست دارد
            # برای سادگی، نتایج را صفر در نظر می‌گیریم یا ادامه نمی‌دهیم.
            # این بخش نیاز به تصمیم‌گیری دقیق‌تر دارد که در صورت خالی بودن داده تست چه اتفاقی بیفتد.
            # فعلا فرض می‌کنیم train_test اجرا نمی‌شود اگر test_data_obj خالی باشد.
            # یک راه حل بهتر: train_test را طوری تغییر دهید که بتواند بدون داده تست اجرا شود (فقط آموزش).
            # یا اگر داده تست خالی است، از بهترین نتایج قبلی استفاده کنید یا خطا دهید.
            # در اینجا، برای جلوگیری از خطا، نتایج را صفر می‌گیریم.
            print("Warning: Test data is empty. Metrics will be 0.")
            recall, mrr, precision = 0,0,0
        else: # داده آموزشی و (داده تست یا داده اعتبارسنجی) وجود دارد
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
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

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
    main()