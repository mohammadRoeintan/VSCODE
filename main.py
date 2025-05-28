# main.py

import argparse
import pickle
import time
import os
import utils # Ensure utils is imported
from model import SessionGraph, train_test, evaluate_model_on_set # evaluate_model_on_set اضافه شده
import torch
import numpy as np
import torch.serialization # Added for add_safe_globals

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty (weight decay)')
parser.add_argument('--step', type=int, default=1, help='gnn propagation steps for local GNN')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='Use validation set splitting')
parser.add_argument('--valid_portion', type=float, default=0.1, help='Portion of training set for validation')
parser.add_argument('--data_subset_ratio', type=float, default=1.0, help='Factor to reduce dataset size (e.g., 0.1 for 10%). Default 1.0.')
parser.add_argument('--ssl_weight', type=float, default=0.1, help='Weight for Self-Supervised Learning Loss')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='Temperature parameter for InfoNCE Loss in SSL')
parser.add_argument('--ssl_dropout_rate', type=float, default=0.2, help='Dropout rate for SSL augmentation')
parser.add_argument('--nhead', type=int, default=2, help='Number of heads in Transformer encoder')
parser.add_argument('--nlayers', type=int, default=2, help='Number of layers in Transformer encoder')
parser.add_argument('--ff_hidden', type=int, default=256, help='Dimension of feedforward network in Transformer')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in Transformer')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training')
parser.add_argument('--global_gcn_layers', type=int, default=1, help='Number of GCN layers for global graph (0 to disable)')
parser.add_argument('--k_metric', type=int, default=20, help='Value of K for Recall@K and MRR@K metrics')

opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    base_dataset_path = './datasets/' # Assuming datasets are in a 'datasets' subdirectory

    if opt.dataset == 'sample':
        train_data_file = os.path.join(base_dataset_path, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, 'test.txt') # مسیر فایل تست اصلی
    else:
        train_data_file = os.path.join(base_dataset_path, opt.dataset, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, opt.dataset, 'test.txt') # مسیر فایل تست اصلی

    # --- Load Full Training Data (for n_node and global graph) ---
    print(f"Loading full training data from: {train_data_file}")
    train_data_loaded_full = None
    try:
        with open(train_data_file, 'rb') as f:
            train_data_loaded_full = pickle.load(f)
        if not (isinstance(train_data_loaded_full, tuple) and len(train_data_loaded_full) == 2 and
                isinstance(train_data_loaded_full[0], list) and isinstance(train_data_loaded_full[1], list)):
            print(f"Error: Full training data at {train_data_file} is not in expected format.")
            return
    except FileNotFoundError:
        print(f"Error: Full training data file not found at {train_data_file}")
        return
    except Exception as e:
        print(f"Error loading or processing full training data: {e}")
        return

    # This will be the actual training data, possibly subsetted
    train_data_for_loader = train_data_loaded_full

    # --- Load Full Test/Validation Data (for n_node) ---
    test_data_loaded_full_for_n_node = ([], []) # Initialize for n_node calculation
    if opt.validation:
        # Validation set will be split from train_data_loaded_full later
        # For n_node calculation, we'll use train_data_loaded_full.
        # test_data_loaded_full_for_n_node remains empty, so test file is NOT used for n_node in validation mode
        pass
    else: # Only load test data for n_node if not using validation split (where test set is used for final eval)
        try:
            with open(test_data_file, 'rb') as f: # test_data_file is the actual test set path
                test_data_loaded_full_for_n_node = pickle.load(f)
            if not (isinstance(test_data_loaded_full_for_n_node, tuple) and \
                    len(test_data_loaded_full_for_n_node) == 2):
                test_data_loaded_full_for_n_node = ([], []) # Reset if format is wrong
        except FileNotFoundError:
            print(f"Warning: Test data file for n_node calculation not found at {test_data_file}.")
        except Exception as e:
            print(f"Warning: Error loading test data for n_node calculation: {e}")


    # --- Calculate n_node (Total number of unique items + 1 for padding) ---
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        all_nodes_for_n_node_calc = set()
        # From full training data
        if train_data_loaded_full and train_data_loaded_full[0]:
            for session in train_data_loaded_full[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes_for_n_node_calc.add(item)
            for target_wrapper in train_data_loaded_full[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes_for_n_node_calc.add(target)
        # From full test/validation data (if available and not opt.validation mode for splitting)
        # This block (test_data_loaded_full_for_n_node) will be empty if opt.validation is true
        if test_data_loaded_full_for_n_node and test_data_loaded_full_for_n_node[0]:
             for session in test_data_loaded_full_for_n_node[0]:
                for item_wrapper in session:
                    item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                    if isinstance(item, int) and item != 0: all_nodes_for_n_node_calc.add(item)
             for target_wrapper in test_data_loaded_full_for_n_node[1]:
                target = target_wrapper[0] if isinstance(target_wrapper, list) and target_wrapper else target_wrapper
                if isinstance(target, int) and target != 0: all_nodes_for_n_node_calc.add(target)

        if not all_nodes_for_n_node_calc:
            print("Critical Error: No nodes found in the dataset(s) to determine n_node.")
            if opt.dataset == 'sample':
                 n_node = 1000 # Fallback for empty sample
                 print(f"Using default n_node = {n_node} for 'sample' dataset.")
            else:
                return
        else:
            n_node = max(all_nodes_for_n_node_calc) + 1
        print(f"Calculated n_node based on dataset: {n_node}")

    if n_node <= 1: # n_node must be at least 2 (item 0 for padding, item 1 for at least one actual item)
        print(f"Error: Invalid n_node ({n_node}). Must be > 1.")
        return

    # --- Data Subsetting (Applied after n_node and global graph source data is determined) ---
    if 0 < opt.data_subset_ratio < 1.0:
        train_sessions, train_targets = train_data_for_loader # Start with full train data
        num_original_train = len(train_sessions)
        if num_original_train > 0:
            num_train_to_keep = int(num_original_train * opt.data_subset_ratio)
            if num_train_to_keep == 0 and num_original_train > 0 : num_train_to_keep = 1 # Keep at least one sample
            train_data_for_loader = (train_sessions[:num_train_to_keep], train_targets[:num_train_to_keep])
            print(f"--- Limiting training data to {len(train_data_for_loader[0])} samples ({opt.data_subset_ratio*100:.1f}%). ---")

    # --- Prepare Data for DataLoaders (train_set_for_data_obj, test_set_for_data_obj / eval_data_obj) ---
    train_set_for_data_obj = train_data_for_loader # This is the (potentially subsetted) training data
    eval_data_source = ([], []) # This will be validation set or test set

    if opt.validation:
        if not train_set_for_data_obj[0]:
            print("Error: Training data is empty, cannot perform validation split.")
            return
        print(f"Splitting training data for validation (portion: {opt.valid_portion}) from {len(train_set_for_data_obj[0])} samples.")
        train_set_for_data_obj, valid_set_for_data_obj = utils.split_validation(train_set_for_data_obj, opt.valid_portion)
        eval_data_source = valid_set_for_data_obj # Use validation set for evaluation during training
        print(f"Number of training samples after split: {len(train_set_for_data_obj[0])}")
        print(f"Number of validation samples: {len(eval_data_source[0])}")
    else:
        # Load actual test data for evaluation (not the one used for n_node if different, though paths are same here)
        try:
            with open(test_data_file, 'rb') as f: # test_data_file is the actual test set path
                eval_data_source = pickle.load(f)
            if not (isinstance(eval_data_source, tuple) and len(eval_data_source) == 2):
                print(f"Error: Test data at {test_data_file} is not in expected format. Using empty for eval.")
                eval_data_source = ([], [])
        except FileNotFoundError:
            print(f"Warning: Test data file for evaluation not found at {test_data_file}. Using empty for eval.")
            eval_data_source = ([], [])
        except Exception as e:
            print(f"Warning: Error loading test data for evaluation: {e}. Using empty for eval.")

        # Apply subsetting to test data if needed (only if not using validation split)
        if 0 < opt.data_subset_ratio < 1.0 and eval_data_source[0]:
            test_sessions, test_targets = eval_data_source
            num_original_test = len(test_sessions)
            if num_original_test > 0:
                num_test_to_keep = int(num_original_test * opt.data_subset_ratio)
                if num_test_to_keep == 0 and num_original_test > 0: num_test_to_keep = 1
                eval_data_source = (test_sessions[:num_test_to_keep], test_targets[:num_test_to_keep])
                print(f"--- Limiting test data (for eval during training) to {len(eval_data_source[0])} samples ({opt.data_subset_ratio*100:.1f}%). ---")
        print(f"Number of samples for evaluation during training (test set): {len(eval_data_source[0])}")


    # --- Build Sparse Global Graph ---
    global_adj_sparse_tensor = None
    if opt.global_gcn_layers > 0:
        # Use full training data (before splitting/subsetting for loader) to build global graph for stability
        source_sessions_for_global_graph = train_data_loaded_full[0] if train_data_loaded_full else []

        if source_sessions_for_global_graph:
            print(f"Building sparse global graph using {len(source_sessions_for_global_graph)} sessions and n_node={n_node}...")
            global_adj_sparse_tensor = utils.build_graph_global_sparse(source_sessions_for_global_graph, n_node)
            if global_adj_sparse_tensor is not None:
                 print(f"Sparse global graph built. Shape: {global_adj_sparse_tensor.shape}, nnz: {global_adj_sparse_tensor._nnz()}")
            else:
                 print("Error: Global graph construction returned None. Disabling global GCN.")
                 opt.global_gcn_layers = 0 # Disable if construction failed
        else:
            print("Warning: No source sessions to build global graph. Disabling global GCN.")
            opt.global_gcn_layers = 0
    else:
        print("Global GCN layers set to 0. Skipping global graph construction.")

    # --- Create Data Objects ---
    train_data_obj = None
    if train_set_for_data_obj and train_set_for_data_obj[0]: # Check if sessions list is not empty
        try:
            train_data_obj = utils.Data(train_set_for_data_obj, shuffle=True)
            print(f"Training Data object created with {train_data_obj.length} samples.")
        except ValueError as e:
            print(f"Error creating Training Data object: {e}. Cannot proceed with training.")
            return
    else:
        print("Training data is empty. Cannot proceed with training.")
        return

    eval_data_obj = None # For test/validation during training loop
    if eval_data_source and eval_data_source[0]: # Check if sessions list is not empty
        try:
            eval_data_obj = utils.Data(eval_data_source, shuffle=False)
            print(f"Evaluation Data object (for training loop) created with {eval_data_obj.length} samples.")
        except ValueError as e:
            print(f"Error creating Evaluation Data object: {e}. Evaluation (during training) will use empty data.")
            eval_data_obj = utils.Data(([], []), shuffle=False) # Fallback to empty
    else:
        print("Evaluation data (for training loop) is empty or not specified. Creating empty Evaluation Data object.")
        eval_data_obj = utils.Data(([], []), shuffle=False) # Fallback to empty

    # --- Initialize Model ---
    model = SessionGraph(opt, n_node, global_adj_sparse_matrix=global_adj_sparse_tensor)
    model.to(device)

    # --- Optimizer and Scheduler ---
    model.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    # --- Checkpoint and Training Loop ---
    checkpoint_dir = f'./checkpoints/{opt.dataset}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    start_epoch = 0
    best_result = [0.0, 0.0, 0.0] # Recall@K, MRR@K, Precision@K
    best_epoch = [0, 0, 0]
    bad_counter = 0

    if opt.resume_from_checkpoint and os.path.exists(opt.resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {opt.resume_from_checkpoint}")
        try:
            custom_safe_globals = [np.ScalarType, np._core.multiarray.scalar]
            with torch.serialization.safe_globals(custom_safe_globals):
                 checkpoint = torch.load(opt.resume_from_checkpoint, map_location=device, weights_only=False)

            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and hasattr(model, 'optimizer'):
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and hasattr(model, 'scheduler'):
                model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint.get('epoch', -1) + 1
            best_result = checkpoint.get('best_result', best_result) # Load previous best result
            best_epoch = checkpoint.get('best_epoch', best_epoch) # Load previous best epoch
            print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}.")
            print(f"Previous best result (Recall@{opt.k_metric}): {best_result[0]:.4f}, MRR@{opt.k_metric}: {best_result[1]:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0 # Reset start_epoch if checkpoint loading fails
            best_result = [0.0, 0.0, 0.0] # Reset best_result
            best_epoch = [0, 0, 0]       # Reset best_epoch
    else:
        if opt.resume_from_checkpoint: # if path was given but file not found
            print(f"Checkpoint file not found: {opt.resume_from_checkpoint}. Starting fresh.")
        else: # No checkpoint path given
            print("No checkpoint specified. Starting fresh.")


    start_time_training = time.time()

    for epoch_num in range(start_epoch, opt.epoch):
        current_lr = model.optimizer.param_groups[0]['lr']
        print(f"{'-'*25} Epoch: {epoch_num} | LR: {current_lr:.6f} {'-'*25}")

        if train_data_obj is None or train_data_obj.length == 0:
            print("Critical Error: train_data_obj is None or empty at training loop. Exiting.")
            return

        recall, mrr, precision = train_test(model, train_data_obj, eval_data_obj, opt)

        flag = 0
        # Only consider improvement and save model if there is actual evaluation data
        if eval_data_obj and eval_data_obj.length > 0:
            if recall >= best_result[0]: # Recall as primary metric
                best_result[0] = recall
                best_epoch[0] = epoch_num
                flag = 1
            if mrr >= best_result[1]: # MRR as secondary metric
                best_result[1] = mrr
                best_epoch[1] = epoch_num
                if not flag : flag = 1 # if MRR improved even if Recall didn't

            # Precision is not currently a primary driver for 'best model' but could be stored
            # best_result[2] = precision
            # best_epoch[2] = epoch_num

            print(f'Current Best (on {"validation" if opt.validation else "test"} set, k={opt.k_metric}): '
                  f'Recall: {best_result[0]:.4f} (Epoch {best_epoch[0]}), '
                  f'MRR: {best_result[1]:.4f} (Epoch {best_epoch[1]})')

            if flag == 1: # If Recall or MRR improved
                best_model_save_path = os.path.join(checkpoint_dir, f'model_best_k{opt.k_metric}.pth')
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'scheduler_state_dict': model.scheduler.state_dict(),
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'opt': opt # Save options for reference
                }, best_model_save_path)
                print(f'Best model checkpoint saved to {best_model_save_path}')
                bad_counter = 0 # Reset bad_counter on improvement
            else:
                bad_counter += 1

            if bad_counter >= opt.patience:
                print(f"Early stopping after {opt.patience} epochs without improvement on {'validation' if opt.validation else 'test'} set.")
                break
        else: # No evaluation data during training loop
            print(f'Epoch {epoch_num} completed (no evaluation data available during training loop).')


    total_training_time = time.time() - start_time_training
    print(f"{'-'*25} Training Finished {'-'*25}")
    print(f"Total Training Run time: {total_training_time:.2f} s")

    if eval_data_obj and eval_data_obj.length > 0:
        eval_set_name = "Validation Set" if opt.validation else "Test Set (used during training)"
        print(f"Final Best Overall Result on {eval_set_name} (k={opt.k_metric}):")
        print(f'\tRecall@{opt.k_metric}: {best_result[0]:.4f} (Achieved at Epoch {best_epoch[0]})')
        print(f'\tMRR@{opt.k_metric}: {best_result[1]:.4f} (Achieved at Epoch {best_epoch[1]})')
    else:
        print("Training finished. No evaluation data was available during training to determine best results.")

    # --- ارزیابی نهایی روی مجموعه تست واقعی ---
    # این بخش تنها زمانی اجرا می‌شود که از مجموعه اعتبارسنجی برای انتخاب مدل استفاده شده باشد
    if opt.validation:
        print(f"\n{'='*30}\nPerforming Final Evaluation on the Designated Test Set\n{'='*30}")

        # ۱. تعیین مسیر فایل داده تست
        final_test_data_file_path = ''
        if opt.dataset == 'sample':
            final_test_data_file_path = os.path.join(base_dataset_path, 'test.txt')
        else:
            final_test_data_file_path = os.path.join(base_dataset_path, opt.dataset, 'test.txt')
        print(f"Attempting to load final test data from: {final_test_data_file_path}")

        # ۲. بارگذاری داده‌های تست
        final_test_set_content = ([], []) # مقدار اولیه
        try:
            with open(final_test_data_file_path, 'rb') as f:
                final_test_set_content = pickle.load(f)
            if not (isinstance(final_test_set_content, tuple) and len(final_test_set_content) == 2 and
                    isinstance(final_test_set_content[0], list) and isinstance(final_test_set_content[1], list)):
                print(f"Error: Final test data at {final_test_data_file_path} is not in the expected format (tuple of two lists). Skipping final test evaluation.")
                final_test_set_content = ([], [])
        except FileNotFoundError:
            print(f"Warning: Final test data file not found at {final_test_data_file_path}. Skipping final test evaluation.")
            final_test_set_content = ([], [])
        except Exception as e:
            print(f"Warning: Error loading final test data: {e}. Skipping final test evaluation.")
            final_test_set_content = ([], [])

        # ۳. اعمال نمونه‌برداری روی داده تست
        if 0 < opt.data_subset_ratio < 1.0 and final_test_set_content[0]:
            test_sessions_final, test_targets_final = final_test_set_content
            num_original_final_test = len(test_sessions_final)
            if num_original_final_test > 0:
                num_final_test_to_keep = int(num_original_final_test * opt.data_subset_ratio)
                if num_final_test_to_keep == 0 and num_original_final_test > 0 : num_final_test_to_keep = 1
                final_test_set_content = (test_sessions_final[:num_final_test_to_keep], test_targets_final[:num_final_test_to_keep])
                print(f"--- Limiting final test data to {len(final_test_set_content[0])} samples ({opt.data_subset_ratio*100:.1f}% of original test set). ---")

        # ۴. ایجاد شیء Data برای مجموعه تست
        final_test_data_object = None
        if final_test_set_content and final_test_set_content[0]:
            try:
                final_test_data_object = utils.Data(final_test_set_content, shuffle=False)
                print(f"Final Test Data object created with {final_test_data_object.length} samples.")
            except ValueError as e:
                print(f"Error creating Final Test Data object: {e}. Skipping final test evaluation.")
        else:
             if opt.validation : # فقط اگر در حالت اعتبارسنجی هستیم این پیام را نشان بده
                 print("Final test data is empty or could not be loaded. Skipping final test evaluation.")


        # ۵. بارگذاری بهترین مدل و ارزیابی
        if final_test_data_object and final_test_data_object.length > 0 :
            best_model_path_for_test = os.path.join(checkpoint_dir, f'model_best_k{opt.k_metric}.pth')
            if os.path.exists(best_model_path_for_test):
                print(f"Loading best model from {best_model_path_for_test} for final evaluation on test set...")

                # n_node و global_adj_sparse_tensor باید از مرحله آموزش در دسترس باشند
                model_for_final_evaluation = SessionGraph(opt, n_node, global_adj_sparse_matrix=global_adj_sparse_tensor)
                model_for_final_evaluation.to(device)

                try:
                    custom_safe_globals_final_eval = [np.ScalarType, np._core.multiarray.scalar]
                    with torch.serialization.safe_globals(custom_safe_globals_final_eval):
                        checkpoint_final_eval = torch.load(best_model_path_for_test, map_location=device, weights_only=False)

                    model_for_final_evaluation.load_state_dict(checkpoint_final_eval['model_state_dict'])
                    print("Best model state loaded successfully for final evaluation on test set.")

                    test_recall, test_mrr, _ = evaluate_model_on_set(
                        model_for_final_evaluation,
                        final_test_data_object,
                        opt,
                        device
                    )

                    print(f"\n{'-'*20} Final Results on ACTUAL TEST SET (using best model from validation) {'-'*20}")
                    print(f'\tRecall@{opt.k_metric}: {test_recall:.4f}%')
                    print(f'\tMRR@{opt.k_metric}: {test_mrr:.4f}%')
                    print(f"{'-'*(40 + len(f' Final Results on ACTUAL TEST SET (using best model from validation) '))}")

                except Exception as e:
                    print(f"An error occurred during final model loading or evaluation on test set: {e}")
            else:
                print(f"Best model checkpoint for final evaluation not found at {best_model_path_for_test}. Cannot perform final test evaluation.")
        elif opt.validation :
            print("Skipping final evaluation on test set as test data was empty or could not be processed.")
    elif not opt.validation: # اگر از ابتدا از مجموعه اعتبارسنجی استفاده نشده بود
        print("\nFinal evaluation on a separate test set is typically performed when a validation set was used for model selection during training.")
        print("Since `opt.validation` was false, the results reported during training were already on the designated test set.")


if __name__ == '__main__':
    main()