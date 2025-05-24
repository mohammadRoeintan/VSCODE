import argparse
import pickle
import time
import os # ماژول os برای کار با مسیرها و دایرکتوری‌ها اضافه شده است
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

# بررسی CUDA بودن دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    # مسیر دیتاست‌ها
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
        if not (isinstance(train_data_loaded, tuple) and len(train_data_loaded) == 2):
            print(f"Error: Training data at {train_data_file} is not in the expected format (sessions, targets).")
            return
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_file}")
        return
    except Exception as e:
        print(f"Error loading or processing training data: {e}")
        return

    if opt.validation:
        train_set, valid_set = split_validation(train_data_loaded, opt.valid_portion)
        test_data_loaded = valid_set
    else:
        print(f"Loading testing data from: {test_data_file}")
        try:
            test_data_loaded = pickle.load(open(test_data_file, 'rb'))
            if not (isinstance(test_data_loaded, tuple) and len(test_data_loaded) == 2):
                print(f"Error: Test data at {test_data_file} is not in the expected format (sessions, targets).")
                return
        except FileNotFoundError:
            print(f"Error: Testing data file not found at {test_data_file}")
            if not opt.validation:
                 print("Error: Test data not found and validation is not enabled.")
                 return
            test_data_loaded = ([], [])
        except Exception as e:
             print(f"Error loading or processing testing data: {e}")
             return

    try:
        train_data_obj = Data(train_set if opt.validation else train_data_loaded, shuffle=True)
        print("Training Data object created successfully.")
    except ValueError as e:
        print(f"Error creating Training Data object: {e}")
        data_to_show = train_set if opt.validation else train_data_loaded
        if isinstance(data_to_show, tuple) and len(data_to_show) == 2:
            print("Sample session (train):", data_to_show[0][0] if data_to_show[0] else "None")
            print("Sample target (train):", data_to_show[1][0] if data_to_show[1] else "None")
        return

    try:
        test_data_obj = Data(test_data_loaded, shuffle=False)
        print("Testing Data object created successfully.")
    except ValueError as e:
        print(f"Error creating Testing Data object: {e}")
        if isinstance(test_data_loaded, tuple) and len(test_data_loaded) == 2:
             print("Sample session (test):", test_data_loaded[0][0] if test_data_loaded[0] else "None")
             print("Sample target (test):", test_data_loaded[1][0] if test_data_loaded[1] else "None")
        return

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        all_nodes = set()
        if train_data_loaded and isinstance(train_data_loaded[0], list):
            for session in train_data_loaded[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in train_data_loaded[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)

        if test_data_loaded and isinstance(test_data_loaded[0], list):
            for session in test_data_loaded[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes.add(item)
            for target_wrapper in test_data_loaded[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes.add(target)

        if not all_nodes:
             print("Warning: No nodes found in the dataset. Setting n_node to a default for 'sample'.")
             n_node = 310 if opt.dataset == 'sample' else 1000
        else:
             n_node = max(all_nodes) + 1
        print(f"Calculated n_node based on actual data: {n_node}")

    if n_node <= 1:
        print(f"Error: Invalid n_node calculated: {n_node}. Check dataset processing.")
        return

    model = SessionGraph(opt, n_node)
    model = trans_to_cuda(model)

    # --- ایجاد پوشه برای ذخیره مدل‌ها ---
    checkpoint_dir = f'./checkpoints/{opt.dataset}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    # ------------------------------------

    start = time.time()
    best_result = [0, 0, 0]
    best_epoch = [0, 0, 0]
    bad_counter = 0

    for epoch_num in range(opt.epoch):
        print(f"{'-'*25} Epoch: {epoch_num} {'-'*25}")
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

        # --- ذخیره مدل بعد از هر اپوک ---
        model_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch_num}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
        # ---------------------------------

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