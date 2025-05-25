import argparse
import pickle
import time
import os
# import random # اگر می‌خواهید نمونه‌برداری تصادفی انجام دهید، این را اضافه کنید
from utils import Data, split_validation
# model.py شامل SessionGraph, train_test است.
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
# --- آرگومان جدید برای محدود کردن داده ---
parser.add_argument('--data_subset_ratio', type=float, default=1.0, help='Factor to reduce dataset size (e.g., 0.1 for 10%). Default is 1.0 (full data).')

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
        train_data_file = os.path.join(base_dataset_path, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, 'test.txt')
    else:
        train_data_file = os.path.join(base_dataset_path, opt.dataset, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, opt.dataset, 'test.txt')

    print(f"Loading training data from: {train_data_file}")
    train_data_loaded_full = None # برای محاسبه n_node بر اساس داده کامل
    try:
        with open(train_data_file, 'rb') as f:
            train_data_loaded_full = pickle.load(f) # ابتدا داده کامل بارگذاری می‌شود
        train_data_loaded = train_data_loaded_full # کپی برای کار و محدودسازی
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
        train_sessions, train_targets = train_data_loaded # استفاده از داده کپی شده
        num_original_train_samples = len(train_sessions)
        if num_original_train_samples > 0:
            num_train_samples_to_keep = int(num_original_train_samples * opt.data_subset_ratio)
            if num_train_samples_to_keep == 0: # حداقل یک نمونه نگه دار اگر داده اصلی وجود دارد
                num_train_samples_to_keep = 1
            
            # یا به سادگی N نمونه اول را بگیرید:
            train_data_loaded = (train_sessions[:num_train_samples_to_keep], train_targets[:num_train_samples_to_keep])
            print(f"--- Limiting training data to {len(train_data_loaded[0])} samples ({opt.data_subset_ratio*100:.1f}%) for testing. ---")
        elif num_original_train_samples == 0: # اگر داده اصلی خالی بود
             print("--- Training data is empty, no subsetting performed. ---")
    # ---------------------------------------------------------

    train_set_for_data_obj = train_data_loaded
    test_data_for_data_obj = ([], [])
    test_data_loaded_full = ([], []) # برای محاسبه n_node بر اساس داده کامل تست

    if opt.validation:
        if not train_data_loaded[0] or not train_data_loaded[1]: # train_data_loaded اینجا ممکن است محدود شده باشد
            print("Error: Training data (potentially subsetted) is empty, cannot perform validation split.")
            return
        print(f"Splitting training data for validation (portion: {opt.valid_portion}) from {len(train_data_loaded[0])} samples.")
        train_set_for_data_obj, valid_set_for_data_obj = split_validation(train_data_loaded, opt.valid_portion)
        test_data_for_data_obj = valid_set_for_data_obj
        # test_data_loaded_full در حالت validation همان valid_set است برای محاسبه n_node
        test_data_loaded_full = valid_set_for_data_obj
        print(f"Number of training samples after split: {len(train_set_for_data_obj[0])}")
        print(f"Number of validation samples: {len(test_data_for_data_obj[0])}")
    else:
        print(f"Loading testing data from: {test_data_file}")
        try:
            with open(test_data_file, 'rb') as f:
                test_data_loaded_full = pickle.load(f) # ابتدا داده کامل تست بارگذاری می‌شود
            test_data_for_data_obj = test_data_loaded_full # کپی برای کار و محدودسازی
            if not (isinstance(test_data_for_data_obj, tuple) and len(test_data_for_data_obj) == 2 and
                    isinstance(test_data_for_data_obj[0], list) and isinstance(test_data_for_data_obj[1], list)):
                print(f"Error: Test data at {test_data_file} is not in the expected format (list_of_sessions, list_of_targets).")
                test_data_for_data_obj = ([], []) # استفاده از داده تست خالی
                test_data_loaded_full = ([], [])
            
            # --- محدود کردن داده تست بر اساس data_subset_ratio ---
            if 0 < opt.data_subset_ratio < 1.0:
                test_sessions, test_targets = test_data_for_data_obj # استفاده از داده کپی شده
                num_original_test_samples = len(test_sessions)
                if num_original_test_samples > 0:
                    num_test_samples_to_keep = int(num_original_test_samples * opt.data_subset_ratio)
                    if num_test_samples_to_keep == 0:
                        num_test_samples_to_keep = 1
                    test_data_for_data_obj = (test_sessions[:num_test_samples_to_keep], test_targets[:num_test_samples_to_keep])
                    print(f"--- Limiting testing data to {len(test_data_for_data_obj[0])} samples ({opt.data_subset_ratio*100:.1f}%) for testing. ---")
                elif num_original_test_samples == 0:
                     print("--- Test data is empty, no subsetting performed. ---")
            # ---------------------------------------------------------

        except FileNotFoundError:
            print(f"Error: Testing data file not found at {test_data_file}. Proceeding with empty test set.")
            test_data_for_data_obj = ([], [])
            test_data_loaded_full = ([], [])
        except Exception as e:
            print(f"Error loading or processing testing data: {e}. Proceeding with empty test set.")
            test_data_for_data_obj = ([], [])
            test_data_loaded_full = ([], [])
        print(f"Number of testing samples to be used: {len(test_data_for_data_obj[0])}")

    # --- محاسبه n_node بر اساس داده‌های کامل اولیه ---
    # این کار تضمین می‌کند n_node صحیح است حتی اگر داده‌ها برای آموزش/تست محدود شده باشند
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        all_nodes = set()
        # استفاده از train_data_loaded_full برای n_node
        if train_data_loaded_full and train_data_loaded_full[0]:
            for session in train_data_loaded_full[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in train_data_loaded_full[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)

        # استفاده از test_data_loaded_full برای n_node (اگر validation نباشد)
        if not opt.validation and test_data_loaded_full and test_data_loaded_full[0]:
            for session in test_data_loaded_full[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in test_data_loaded_full[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)
        # اگر opt.validation بود، test_data_loaded_full قبلاً با valid_set مقداردهی شده
        # که خود بخشی از train_data_loaded_full است، پس آیتم‌های آن در all_nodes لحاظ شده‌اند.

        if not all_nodes:
             print("Critical Error: No nodes found in the full dataset. Cannot determine n_node.")
             return
        else:
             n_node = max(all_nodes) + 1
        print(f"Calculated n_node based on FULL dataset: {n_node}")

    if n_node <= 1:
        print(f"Error: Invalid n_node calculated: {n_node}. Must be > 1.")
        return

    # --- ساخت آبجکت‌های Data با داده‌های نهایی (احتمالاً محدود شده) ---
    train_data_obj = None
    test_data_obj_for_eval = None # برای ارزیابی

    try:
        if train_set_for_data_obj[0]: # فقط اگر داده آموزشی وجود دارد
            train_data_obj = Data(train_set_for_data_obj, shuffle=True)
            print(f"Training Data object created successfully with {len(train_data_obj.inputs)} samples.")
        else:
            print("Training data is empty. Training Data object not created.")
    except ValueError as e:
        print(f"Error creating Training Data object: {e}")
        return

    try:
        if test_data_for_data_obj[0]: # فقط اگر داده تست/اعتبارسنجی وجود دارد
            test_data_obj_for_eval = Data(test_data_for_data_obj, shuffle=False)
            print(f"Test/Validation Data object created successfully with {len(test_data_obj_for_eval.inputs)} samples.")
        else:
            print("Test/Validation data is empty. Test/Validation Data object not created.")
    except ValueError as e:
        print(f"Error creating Test/Validation Data object: {e}")
        return
    # -------------------------------------------------------------

    model = SessionGraph(opt, n_node)
    model.to(device)

    checkpoint_dir = f'/checkpoints/{opt.dataset}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    start = time.time()
    best_result = [0, 0, 0]
    best_epoch = [0, 0, 0]
    bad_counter = 0

    for epoch_num in range(opt.epoch):
        print(f"{'-'*25} Epoch: {epoch_num} {'-'*25}")

        if train_data_obj is None or train_data_obj.length == 0:
            print(f"Epoch {epoch_num}: No training data. Skipping epoch.")
            # اگر داده آموزشی نباشد، ادامه آموزش معنی ندارد
            if epoch_num == 0 : # اگر از ابتدا داده آموزشی نباشد، متوقف شو
                 print("Exiting: No training data available from the start.")
                 return
            continue # اگر در اپوک‌های بعدی داده آموزشی از بین رفته (نباید اتفاق بیفتد)

        # اجرای آموزش و تست
        # اگر test_data_obj_for_eval خالی باشد، train_test باید بتواند این حالت را مدیریت کند
        # یا باید اینجا از اجرای آن صرف نظر کرد.
        # فرض می‌کنیم train_test می‌تواند با test_data خالی اجرا شود و فقط بخش آموزش را انجام دهد
        # یا نتایج ارزیابی را صفر برگرداند. (این باید در model.py بررسی شود)
        if test_data_obj_for_eval is None or test_data_obj_for_eval.length == 0:
            print(f"Epoch {epoch_num}: No test/validation data for evaluation. Metrics will be 0 or based on dummy evaluation if any.")
            # اگر train_test نیاز به test_data دارد، باید اینجا یک dummy test_data ایجاد کرد یا train_test را تغییر داد
            # برای سادگی، یک test_data_obj خالی (اما معتبر) می‌سازیم اگر وجود ندارد
            dummy_test_data_obj = Data(([],[]), shuffle=False) if test_data_obj_for_eval is None else test_data_obj_for_eval
            recall, mrr, precision = train_test(model, train_data_obj, dummy_test_data_obj, opt)
            if test_data_obj_for_eval is None or test_data_obj_for_eval.length == 0:
                recall, mrr, precision = 0.0, 0.0, 0.0 # بازنویسی نتایج اگر واقعا داده تست نبود
        else:
            recall, mrr, precision = train_test(model, train_data_obj, test_data_obj_for_eval, opt)


        flag = 0
        # فقط اگر داده تست/اعتبارسنجی وجود داشت، نتایج را آپدیت کن و early stopping را بررسی کن
        if test_data_obj_for_eval and test_data_obj_for_eval.length > 0:
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
            print('Current Best Result (on available evaluation data):')
            print(f'\tRecall@20: {best_result[0]:.4f}\tMRR@20: {best_result[1]:.4f}\tPrecision@20: {best_result[2]:.4f}\tEpochs: ({best_epoch[0]}, {best_epoch[1]}, {best_epoch[2]})')
            bad_counter += (1 - flag)
            if bad_counter >= opt.patience:
                print(f"Early stopping triggered after {opt.patience} epochs without improvement.")
                break
        else: # اگر داده ارزیابی نبود، فقط نتایج اپوک فعلی (که صفر خواهد بود) را چاپ کن
            print(f'Epoch {epoch_num} completed (no evaluation data). Recall: {recall:.4f}, MRR: {mrr:.4f}, Precision: {precision:.4f}')


        model_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch_num}.pth')
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'scheduler_state_dict': model.scheduler.state_dict(),
            'best_result': best_result,
            'opt': opt
        }, model_save_path)
        print(f'Model checkpoint saved to {model_save_path}')


    print(f"{'-'*25} Training Finished {'-'*25}")
    end = time.time()
    print(f"Total Run time: {end - start:.2f} s")
    if test_data_obj_for_eval and test_data_obj_for_eval.length > 0:
        print("Final Best Overall Result (on available evaluation data):")
        print(f'\tRecall@20: {best_result[0]:.4f}\tMRR@20: {best_result[1]:.4f}\tPrecision@20: {best_result[2]:.4f}\tAchieved at Epochs: ({best_epoch[0]}, {best_epoch[1]}, {best_epoch[2]})')
    else:
        print("Training finished (no evaluation data was available to determine best results).")

if __name__ == '__main__':
    main()