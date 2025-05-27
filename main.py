# main.py
# ... (بخش importها و ArgumentParser مثل قبل، فقط utils را چک کنید که درست import شده باشد)
import argparse
import pickle
import time
import os
import utils # تغییر برای فراخوانی utils.build_graph_global_sparse
from model import SessionGraph, train_test
import torch
import numpy as np
import torch.serialization

# ... (آرگومان‌ها مثل قبل) ...
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps for local GNN')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--data_subset_ratio', type=float, default=1.0, help='Factor to reduce dataset size. Default is 1.0 (full data).')
parser.add_argument('--ssl_weight', type=float, default=0.1, help='Weight for Self-Supervised Learning Loss')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='Temperature parameter for InfoNCE Loss')
parser.add_argument('--ssl_dropout_rate', type=float, default=0.2, help='Dropout rate for SSL augmentation')
parser.add_argument('--nhead', type=int, default=2, help='number of heads in transformer encoder')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers in transformer encoder')
parser.add_argument('--ff_hidden', type=int, default=256, help='dimension of feedforward network model in transformer')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in transformer')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to the checkpoint file to resume training from.')
parser.add_argument('--global_gcn_layers', type=int, default=1, help='Number of GCN layers for global graph (0 to disable).')
parser.add_argument('--k_metric', type=int, default=20, help='Value of K for Recall@K and MRR@K metrics.')


opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    # ... (بخش بارگذاری داده‌ها و محاسبه n_node مثل قبل) ...
    base_dataset_path = './datasets/'

    if opt.dataset == 'sample':
        train_data_file = os.path.join(base_dataset_path, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, 'test.txt')
    else:
        train_data_file = os.path.join(base_dataset_path, opt.dataset, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, opt.dataset, 'test.txt')

    print(f"Loading training data from: {train_data_file}")
    train_data_loaded_full = None
    try:
        with open(train_data_file, 'rb') as f:
            train_data_loaded_full = pickle.load(f)
        train_data_loaded = train_data_loaded_full
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

    if 0 < opt.data_subset_ratio < 1.0:
        train_sessions, train_targets = train_data_loaded
        num_original_train_samples = len(train_sessions)
        if num_original_train_samples > 0:
            num_train_samples_to_keep = int(num_original_train_samples * opt.data_subset_ratio)
            if num_train_samples_to_keep == 0:
                num_train_samples_to_keep = 1 # حداقل یک نمونه برای تست
            train_data_loaded = (train_sessions[:num_train_samples_to_keep], train_targets[:num_train_samples_to_keep])
            print(f"--- Limiting training data to {len(train_data_loaded[0])} samples ({opt.data_subset_ratio*100:.1f}%) for testing. ---")
        elif num_original_train_samples == 0:
             print("--- Training data is empty, no subsetting performed. ---")

    train_set_for_data_obj = train_data_loaded
    test_data_for_data_obj = ([], []) # مقداردهی اولیه
    test_data_loaded_full = ([], []) # برای n_node

    if opt.validation:
        if not train_data_loaded[0] or not train_data_loaded[1]: # بررسی train_data_loaded قبل از تقسیم
            print("Error: Training data (potentially subsetted) is empty, cannot perform validation split.")
            return
        print(f"Splitting training data for validation (portion: {opt.valid_portion}) from {len(train_data_loaded[0])} samples.")
        train_set_for_data_obj, valid_set_for_data_obj = utils.split_validation(train_data_loaded, opt.valid_portion)
        test_data_for_data_obj = valid_set_for_data_obj
        test_data_loaded_full = valid_set_for_data_obj # برای n_node در حالت ولیدیشن
        print(f"Number of training samples after split: {len(train_set_for_data_obj[0])}")
        print(f"Number of validation samples: {len(test_data_for_data_obj[0])}")
    else:
        print(f"Loading testing data from: {test_data_file}")
        try:
            with open(test_data_file, 'rb') as f:
                test_data_loaded_full = pickle.load(f) # برای n_node
            test_data_for_data_obj = test_data_loaded_full # برای استفاده در Data object
            if not (isinstance(test_data_for_data_obj, tuple) and len(test_data_for_data_obj) == 2 and
                    isinstance(test_data_for_data_obj[0], list) and isinstance(test_data_for_data_obj[1], list)):
                print(f"Error: Test data at {test_data_file} is not in the expected format. Proceeding with empty test set for evaluation.")
                test_data_for_data_obj = ([], []) # برای Data object
                # test_data_loaded_full را برای n_node نگه می‌داریم اگر فرمت اشتباه باشد
            
            if 0 < opt.data_subset_ratio < 1.0 and test_data_for_data_obj[0]: # فقط اگر داده تست معتبر وجود دارد
                test_sessions, test_targets = test_data_for_data_obj
                num_original_test_samples = len(test_sessions)
                if num_original_test_samples > 0:
                    num_test_samples_to_keep = int(num_original_test_samples * opt.data_subset_ratio)
                    if num_test_samples_to_keep == 0:
                        num_test_samples_to_keep = 1
                    test_data_for_data_obj = (test_sessions[:num_test_samples_to_keep], test_targets[:num_test_samples_to_keep])
                    print(f"--- Limiting testing data to {len(test_data_for_data_obj[0])} samples ({opt.data_subset_ratio*100:.1f}%) for testing. ---")
                elif num_original_test_samples == 0:
                     print("--- Test data is empty, no subsetting performed (already empty). ---")

        except FileNotFoundError:
            print(f"Error: Testing data file not found at {test_data_file}. Proceeding with empty test set for evaluation.")
            test_data_for_data_obj = ([], [])
            test_data_loaded_full = ([], []) # برای n_node
        except Exception as e:
            print(f"Error loading or processing testing data: {e}. Proceeding with empty test set for evaluation.")
            test_data_for_data_obj = ([], [])
            test_data_loaded_full = ([], []) # برای n_node
        print(f"Number of testing samples to be used for evaluation: {len(test_data_for_data_obj[0])}")


    # محاسبه n_node (مثل قبل، اطمینان از استفاده از داده‌های کامل قبل از subsetting)
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else: # برای 'sample' یا هر دیتاست دیگری
        all_nodes_for_n_node_calc = set()
        # استفاده از train_data_loaded_full
        if train_data_loaded_full and train_data_loaded_full[0]:
            for session in train_data_loaded_full[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes_for_n_node_calc.add(item)
            for target_wrapper in train_data_loaded_full[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes_for_n_node_calc.add(target)

        # استفاده از test_data_loaded_full (که یا از فایل تست خوانده شده یا داده ولیدیشن است)
        if test_data_loaded_full and test_data_loaded_full[0]:
            for session in test_data_loaded_full[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes_for_n_node_calc.add(item)
            for target_wrapper in test_data_loaded_full[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes_for_n_node_calc.add(target)
        
        if not all_nodes_for_n_node_calc:
             print("Critical Error: No nodes found in the full dataset (train/test). Cannot determine n_node.")
             if opt.dataset == 'sample': # یک مقدار پیش‌فرض برای sample اگر واقعا خالی باشد
                 n_node = 1000 # یا یک مقدار مناسب دیگر برای دیتاست نمونه شما
                 print(f"Using default n_node = {n_node} for 'sample' dataset as it appears empty.")
             else:
                return # خروج اگر n_node قابل محاسبه نباشد
        else:
             n_node = max(all_nodes_for_n_node_calc) + 1
        print(f"Calculated n_node based on FULL dataset: {n_node}")

    if n_node <= 1:
        print(f"Error: Invalid n_node calculated: {n_node}. Must be > 1.")
        return

    # --- ساخت گراف گلوبال پراکنده ---
    global_adj_sparse_tensor = None
    if opt.global_gcn_layers > 0:
        all_sessions_for_global_graph_build = []
        if train_data_loaded_full and train_data_loaded_full[0]: # استفاده از داده آموزشی کامل
            all_sessions_for_global_graph_build.extend(train_data_loaded_full[0])
        # تصمیم بگیرید که آیا داده تست هم در ساخت گراف گلوبال نقش داشته باشد یا خیر
        # if test_data_loaded_full and test_data_loaded_full[0] and not opt.validation:
        #    all_sessions_for_global_graph_build.extend(test_data_loaded_full[0])
            
        if all_sessions_for_global_graph_build:
            print(f"Building sparse global graph using {len(all_sessions_for_global_graph_build)} sessions and n_node={n_node}...")
            # تابع جدید utils.build_graph_global_sparse یک تنسور پراکنده PyTorch برمی‌گرداند
            global_adj_sparse_tensor = utils.build_graph_global_sparse(all_sessions_for_global_graph_build, n_node)
            print(f"Sparse global graph built and normalized. Shape: {global_adj_sparse_tensor.shape}, nnz: {global_adj_sparse_tensor._nnz()}")
        else:
            print("Warning: No sessions available to build a global graph. Global GCN will be disabled.")
            opt.global_gcn_layers = 0 # غیرفعال کردن GCN گلوبال
    else:
        print("Global GCN layers set to 0 by user. Skipping global graph construction.")
    # --- پایان ساخت گراف گلوبال ---

    # ایجاد Data objectها (مثل قبل)
    train_data_obj = None
    test_data_obj_for_eval = None

    try:
        if train_set_for_data_obj[0]: # train_set_for_data_obj همان train_data_loaded یا بخش آموزشی از split است
            train_data_obj = utils.Data(train_set_for_data_obj, shuffle=True)
            print(f"Training Data object created with {train_data_obj.length} samples.")
        else:
            print("Training data is empty. Training Data object not created.")
    except ValueError as e:
        print(f"Error creating Training Data object: {e}")
        return

    try:
        if test_data_for_data_obj[0]: # test_data_for_data_obj داده تست یا ولیدیشن است
            test_data_obj_for_eval = utils.Data(test_data_for_data_obj, shuffle=False)
            print(f"Test/Validation Data object created with {test_data_obj_for_eval.length} samples.")
        else:
            print("Test/Validation data is empty. Test/Validation Data object not created.")
            # اگر داده ارزیابی خالی است، یک Data object خالی ایجاد می‌کنیم تا train_test خطا ندهد
            test_data_obj_for_eval = utils.Data(([], []), shuffle=False)

    except ValueError as e:
        print(f"Error creating Test/Validation Data object: {e}")
        # اگر خطا در ایجاد test_data_obj رخ داد، با یک شیء خالی ادامه می‌دهیم
        test_data_obj_for_eval = utils.Data(([], []), shuffle=False)
        print("Proceeding with empty evaluation data due to error.")

    # پاس دادن ماتریس پراکنده گلوبال به مدل
    model = SessionGraph(opt, n_node, global_adj_sparse_matrix=global_adj_sparse_tensor)
    model.to(device) # مدل و بافرهای آن (از جمله گراف گلوبال پراکنده) به دستگاه منتقل می‌شوند
    # ساخت Optimizer و Scheduler و اتصال به مدل
    model.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    # ... (بخش چک‌پوینت و حلقه آموزش مثل قبل، با این تفاوت که k_metric از opt خوانده می‌شود) ...
    checkpoint_dir = f'./checkpoints/{opt.dataset}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    start_epoch = 0
    best_result = [0, 0, 0] # Recall@K, MRR@K, Precision@K (Precision فعلا استفاده نمی‌شود)
    best_epoch = [0, 0, 0]
    bad_counter = 0

    if opt.resume_from_checkpoint and os.path.exists(opt.resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {opt.resume_from_checkpoint}")
        try:
            custom_safe_globals = [np.ScalarType, np._core.multiarray.scalar] # برای سازگاری با pickle numpy
            # اطمینان از اینکه torch.serialization.add_safe_globals فقط یک بار فراخوانی می‌شود یا از context manager استفاده شود
            # torch.serialization.add_safe_globals([np.ScalarType, np._core.multiarray.scalar]) # این ممکن است全局 باشد
            with torch.serialization.safe_globals(custom_safe_globals):
                 checkpoint = torch.load(opt.resume_from_checkpoint, map_location=device, weights_only=False) # weights_only=False برای بارگذاری optimizer و scheduler
            
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and hasattr(model, 'optimizer'):
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and hasattr(model, 'scheduler'):
                model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', -1) + 1 # .get برای سازگاری، اگر epoch ذخیره نشده باشد
            best_result = checkpoint.get('best_result', best_result) # مقدار پیش‌فرض اگر best_result ذخیره نشده
            # opt_loaded = checkpoint.get('opt', opt) # با احتیاط استفاده شود، ممکن است تنظیمات فعلی را بازنویسی کند
            print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}.")
            print(f"Previous best result: Recall@{opt.k_metric}: {best_result[0]:.4f}, MRR@{opt.k_metric}: {best_result[1]:.4f}")

        except RuntimeError as e: # اگر state_dict با مدل فعلی نخواند
            print(f"Error loading checkpoint state_dict: {e}. Starting training from scratch.")
            start_epoch = 0
        except Exception as e: # خطاهای دیگر مثل pickle یا فایل
            print(f"An unexpected error occurred while loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
    else:
        if opt.resume_from_checkpoint: # اگر مسیر داده شده ولی فایل وجود ندارد
            print(f"Checkpoint file not found at {opt.resume_from_checkpoint}. Starting training from scratch.")
        else: # اگر هیچ چک‌پوینتی مشخص نشده
            print("No checkpoint specified. Starting training from scratch.")


    start_time = time.time()

    for epoch_num in range(start_epoch, opt.epoch):
        print(f"{'-'*25} Epoch: {epoch_num} {'-'*25}")

        if train_data_obj is None or train_data_obj.length == 0:
            print(f"Epoch {epoch_num}: No training data. Skipping epoch.")
            if epoch_num == 0 and start_epoch == 0: # اگر از ابتدا داده آموزشی نباشد
                 print("Exiting: No training data available from the start.")
                 return
            continue # اگر در ادامه داده آموزشی از بین برود

        # تابع train_test هم آموزش و هم ارزیابی را انجام می‌دهد
        # test_data_obj_for_eval از قبل با یک Data object خالی مقداردهی شده اگر داده تست واقعی نباشد
        recall, mrr, precision = train_test(model, train_data_obj, test_data_obj_for_eval, opt)
        
        flag = 0
        # فقط اگر داده ارزیابی معتبر وجود دارد (نه Data object خالی ساخته شده)، بهترین نتایج را به‌روزرسانی کن
        if test_data_obj_for_eval and test_data_obj_for_eval.length > 0:
            # فرض: recall اولین، mrr دومین متریک اصلی هستند
            if recall >= best_result[0]:
                best_result[0] = recall
                best_epoch[0] = epoch_num
                flag = 1
            if mrr >= best_result[1]:
                best_result[1] = mrr
                best_epoch[1] = epoch_num
                flag = 1
            # Precision فعلا پایش نمی‌شود، اما می‌توان اضافه کرد
            # if precision >= best_result[2]:
            #     best_result[2] = precision
            #     best_epoch[2] = epoch_num
            #     flag = 1

            print(f'Current Best Result (on available evaluation data for k={opt.k_metric}):')
            print(f'\tRecall@{opt.k_metric}: {best_result[0]:.4f}\tMRR@{opt.k_metric}: {best_result[1]:.4f}\tEpochs: ({best_epoch[0]}, {best_epoch[1]})')
            
            if flag == 1: # اگر هر یک از بهترین نتایج بهبود یافت
                best_model_save_path = os.path.join(checkpoint_dir, f'model_best_k{opt.k_metric}.pth') # نام فایل با k
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'scheduler_state_dict': model.scheduler.state_dict(),
                    'best_result': best_result, # شامل recall, mrr, precision
                    'opt': opt # ذخیره تنظیمات برای بازتولیدپذیری
                }, best_model_save_path)
                print(f'Best model checkpoint saved to {best_model_save_path}')

            bad_counter += (1 - flag)
            if bad_counter >= opt.patience:
                print(f"Early stopping triggered after {opt.patience} epochs without improvement on primary metrics.")
                break
        else: # اگر داده ارزیابی وجود ندارد
            print(f'Epoch {epoch_num} completed (no evaluation data). Training metrics (approx): Recall: {recall:.4f}, MRR: {mrr:.4f}')


    print(f"{'-'*25} Training Finished {'-'*25}")
    end_time = time.time()
    print(f"Total Run time: {end_time - start_time:.2f} s")

    if test_data_obj_for_eval and test_data_obj_for_eval.length > 0:
        print(f"Final Best Overall Result (on available evaluation data for k={opt.k_metric}):")
        print(f'\tRecall@{opt.k_metric}: {best_result[0]:.4f}\tMRR@{opt.k_metric}: {best_result[1]:.4f}\tAchieved at Epochs: ({best_epoch[0]}, {best_epoch[1]})')
    else:
        print("Training finished (no evaluation data was available to determine best results).")


if __name__ == '__main__':
    main()