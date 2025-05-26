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
parser.add_argument('--data_subset_ratio', type=float, default=1.0, help='Factor to reduce dataset size (e.g., 0.1 for 10%). Default is 1.0 (full data).')
parser.add_argument('--ssl_weight', type=float, default=0.1, help='Weight for Self-Supervised Learning Loss')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='Temperature parameter for InfoNCE Loss')
parser.add_argument('--ssl_dropout_rate', type=float, default=0.2, help='Dropout rate for SSL augmentation')
parser.add_argument('--nhead', type=int, default=2, help='number of heads in transformer encoder')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers in transformer encoder')
parser.add_argument('--ff_hidden', type=int, default=256, help='dimension of feedforward network model in transformer')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in transformer')

# --- آرگومان جدید برای بارگذاری مدل ذخیره شده ---
parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from.')

opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    base_dataset_path = './datasets/'
    start_epoch = 0 # اپوک شروع به صورت پیش‌فرض 0 است
    best_result = [0, 0, 0] # Recall, MRR, Precision
    best_epoch = [0, 0, 0] # اپوک‌هایی که بهترین نتایج در آن‌ها به دست آمده

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
                num_train_samples_to_keep = 1
            train_data_loaded = (train_sessions[:num_train_samples_to_keep], train_targets[:num_train_samples_to_keep])
            print(f"--- Limiting training data to {len(train_data_loaded[0])} samples ({opt.data_subset_ratio*100:.1f}%) for testing. ---")
        elif num_original_train_samples == 0:
             print("--- Training data is empty, no subsetting performed. ---")

    train_set_for_data_obj = train_data_loaded
    test_data_for_data_obj = ([], [])
    test_data_loaded_full = ([], [])

    if opt.validation:
        if not train_data_loaded[0] or not train_data_loaded[1]:
            print("Error: Training data (potentially subsetted) is empty, cannot perform validation split.")
            return
        print(f"Splitting training data for validation (portion: {opt.valid_portion}) from {len(train_data_loaded[0])} samples.")
        train_set_for_data_obj, valid_set_for_data_obj = split_validation(train_data_loaded, opt.valid_portion)
        test_data_for_data_obj = valid_set_for_data_obj
        test_data_loaded_full = valid_set_for_data_obj
        print(f"Number of training samples after split: {len(train_set_for_data_obj[0])}")
        print(f"Number of validation samples: {len(test_data_for_data_obj[0])}")
    else:
        print(f"Loading testing data from: {test_data_file}")
        try:
            with open(test_data_file, 'rb') as f:
                test_data_loaded_full = pickle.load(f)
            test_data_for_data_obj = test_data_loaded_full
            if not (isinstance(test_data_for_data_obj, tuple) and len(test_data_for_data_obj) == 2 and
                    isinstance(test_data_for_data_obj[0], list) and isinstance(test_data_for_data_obj[1], list)):
                print(f"Error: Test data at {test_data_file} is not in the expected format (list_of_sessions, list_of_targets).")
                test_data_for_data_obj = ([], [])
                test_data_loaded_full = ([], [])
            
            if 0 < opt.data_subset_ratio < 1.0:
                test_sessions, test_targets = test_data_for_data_obj
                num_original_test_samples = len(test_sessions)
                if num_original_test_samples > 0:
                    num_test_samples_to_keep = int(num_original_test_samples * opt.data_subset_ratio)
                    if num_test_samples_to_keep == 0:
                        num_test_samples_to_keep = 1
                    test_data_for_data_obj = (test_sessions[:num_test_samples_to_keep], test_targets[:num_test_samples_to_keep])
                    print(f"--- Limiting testing data to {len(test_data_for_data_obj[0])} samples ({opt.data_subset_ratio*100:.1f}%) for testing. ---")
                elif num_original_test_samples == 0:
                     print("--- Test data is empty, no subsetting performed. ---")

        except FileNotFoundError:
            print(f"Error: Testing data file not found at {test_data_file}. Proceeding with empty test set.")
            test_data_for_data_obj = ([], [])
            test_data_loaded_full = ([], [])
        except Exception as e:
            print(f"Error loading or processing testing data: {e}. Proceeding with empty test set.")
            test_data_for_data_obj = ([], [])
            test_data_loaded_full = ([], [])
        print(f"Number of testing samples to be used: {len(test_data_for_data_obj[0])}")

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        all_nodes = set()
        if train_data_loaded_full and train_data_loaded_full[0]:
            for session in train_data_loaded_full[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in train_data_loaded_full[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)

        if not opt.validation and test_data_loaded_full and test_data_loaded_full[0]:
            for session in test_data_loaded_full[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in test_data_loaded_full[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)
        
        if not all_nodes:
             print("Critical Error: No nodes found in the full dataset. Cannot determine n_node.")
             return
        else:
             n_node = max(all_nodes) + 1
        print(f"Calculated n_node based on FULL dataset: {n_node}")

    if n_node <= 1:
        print(f"Error: Invalid n_node calculated: {n_node}. Must be > 1.")
        return

    train_data_obj = None
    test_data_obj_for_eval = None

    try:
        if train_set_for_data_obj[0]:
            train_data_obj = Data(train_set_for_data_obj, shuffle=True)
            print(f"Training Data object created successfully with {len(train_data_obj.inputs)} samples.")
        else:
            print("Training data is empty. Training Data object not created.")
    except ValueError as e:
        print(f"Error creating Training Data object: {e}")
        return

    try:
        if test_data_for_data_obj[0]:
            test_data_obj_for_eval = Data(test_data_for_data_obj, shuffle=False)
            print(f"Test/Validation Data object created successfully with {len(test_data_obj_for_eval.inputs)} samples.")
        else:
            print("Test/Validation data is empty. Test/Validation Data object not created.")
    except ValueError as e:
        print(f"Error creating Test/Validation Data object: {e}")
        return

    model = SessionGraph(opt, n_node)
    model.to(device)

    checkpoint_dir = f'./checkpoints/{opt.dataset}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    # --- بارگذاری Checkpoint در صورت نیاز ---
    if opt.resume_checkpoint and os.path.exists(opt.resume_checkpoint):
        print(f"Resuming training from checkpoint: {opt.resume_checkpoint}")
        try:
            checkpoint = torch.load(opt.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint.get('best_result', [0, 0, 0]) # مقدار پیش‌فرض اگر در checkpoint قدیمی نباشد
            best_epoch = checkpoint.get('best_epoch', [0, 0, 0])   # مقدار پیش‌فرض
            # opt_checkpoint = checkpoint.get('opt', None) # می‌توان برای بررسی سازگاری استفاده کرد
            print(f"Loaded checkpoint. Resuming from epoch {start_epoch}.")
            print(f"Previous best results: Recall@{best_epoch[0]}={best_result[0]:.4f}, MRR@{best_epoch[1]}={best_result[1]:.4f}, Precision@{best_epoch[2]}={best_result[2]:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
            best_result = [0,0,0]
            best_epoch = [0,0,0]
    else:
        if opt.resume_checkpoint:
             print(f"Checkpoint file not found at {opt.resume_checkpoint}. Starting training from scratch.")
        else:
             print("No checkpoint specified. Starting training from scratch.")
    # -------------------------------------

    start_time = time.time()
    bad_counter = 0

    for epoch_num in range(start_epoch, opt.epoch):
        print(f"{'-'*25} Epoch: {epoch_num} {'-'*25}")

        if train_data_obj is None or train_data_obj.length == 0:
            print(f"Epoch {epoch_num}: No training data. Skipping epoch.")
            if epoch_num == 0 :
                 print("Exiting: No training data available from the start.")
                 return
            continue

        if test_data_obj_for_eval is None or test_data_obj_for_eval.length == 0:
            print(f"Epoch {epoch_num}: No test/validation data for evaluation. Metrics will be 0 or based on dummy evaluation if any.")
            dummy_test_data_obj = Data(([],[]), shuffle=False) if test_data_obj_for_eval is None else test_data_obj_for_eval
            recall, mrr, precision = train_test(model, train_data_obj, dummy_test_data_obj, opt)
            if test_data_obj_for_eval is None or test_data_obj_for_eval.length == 0:
                recall, mrr, precision = 0.0, 0.0, 0.0
        else:
            recall, mrr, precision = train_test(model, train_data_obj, test_data_obj_for_eval, opt)

        flag = 0
        if test_data_obj_for_eval and test_data_obj_for_eval.length > 0:
            if recall >= best_result[0]:
                best_result[0] = recall
                best_epoch[0] = epoch_num
                flag = 1
                # ذخیره مدل با بهترین Recall
                best_recall_model_path = os.path.join(checkpoint_dir, f'model_best_recall.pth')
                print(f"New best recall: {recall:.4f}. Saving model to {best_recall_model_path}")
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'scheduler_state_dict': model.scheduler.state_dict(),
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'opt': opt
                }, best_recall_model_path)

            if mrr >= best_result[1]:
                best_result[1] = mrr
                best_epoch[1] = epoch_num
                flag = 1
                # ذخیره مدل با بهترین MRR
                best_mrr_model_path = os.path.join(checkpoint_dir, f'model_best_mrr.pth')
                print(f"New best MRR: {mrr:.4f}. Saving model to {best_mrr_model_path}")
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'scheduler_state_dict': model.scheduler.state_dict(),
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'opt': opt
                }, best_mrr_model_path)

            if precision >= best_result[2]:
                best_result[2] = precision
                best_epoch[2] = epoch_num
                flag = 1
                # ذخیره مدل با بهترین Precision
                best_precision_model_path = os.path.join(checkpoint_dir, f'model_best_precision.pth')
                print(f"New best precision: {precision:.4f}. Saving model to {best_precision_model_path}")
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'scheduler_state_dict': model.scheduler.state_dict(),
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'opt': opt
                }, best_precision_model_path)

            print('Current Best Result (on available evaluation data):')
            print(f'\tRecall@20: {best_result[0]:.4f} (Epoch {best_epoch[0]})\tMRR@20: {best_result[1]:.4f} (Epoch {best_epoch[1]})\tPrecision@20: {best_result[2]:.4f} (Epoch {best_epoch[2]})')
            
            bad_counter += (1 - flag)
            if bad_counter >= opt.patience:
                print(f"Early stopping triggered after {opt.patience} epochs without improvement in any metric.")
                break
        else:
            print(f'Epoch {epoch_num} completed (no evaluation data). Recall: {recall:.4f}, MRR: {mrr:.4f}, Precision: {precision:.4f}')

        # --- ذخیره سازی جامع Checkpoint در هر اپوک ---
        model_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch_num}.pth')
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'scheduler_state_dict': model.scheduler.state_dict(),
            'best_result': best_result, # ذخیره بهترین نتایج تا این اپوک
            'best_epoch': best_epoch,   # ذخیره اپوک‌های بهترین نتایج تا این اپوک
            'current_metrics': {'recall': recall, 'mrr': mrr, 'precision': precision}, # نتایج این اپوک
            'opt': opt # ذخیره تنظیمات استفاده شده برای این اجرا
        }, model_save_path)
        print(f'Model checkpoint for epoch {epoch_num} saved to {model_save_path}')
        # -------------------------------------------

    print(f"{'-'*25} Training Finished {'-'*25}")
    end_time = time.time()
    print(f"Total Run time: {end_time - start_time:.2f} s")
    if test_data_obj_for_eval and test_data_obj_for_eval.length > 0:
        print("Final Best Overall Result (on available evaluation data):")
        print(f'\tRecall@20: {best_result[0]:.4f} (Achieved at Epoch {best_epoch[0]})')
        print(f'\tMRR@20: {best_result[1]:.4f} (Achieved at Epoch {best_epoch[1]})')
        print(f'\tPrecision@20: {best_result[2]:.4f} (Achieved at Epoch {best_epoch[2]})')
    else:
        print("Training finished (no evaluation data was available to determine best results).")

if __name__ == '__main__':
    main()