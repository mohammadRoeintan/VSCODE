# main.py

import argparse
import pickle
import time
import os
import utils # Ensure utils is imported
from model import SessionGraph, train_test # model.py includes SessionGraph, train_test
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
        test_data_file = os.path.join(base_dataset_path, 'test.txt')
    else:
        train_data_file = os.path.join(base_dataset_path, opt.dataset, 'train.txt')
        test_data_file = os.path.join(base_dataset_path, opt.dataset, 'test.txt')

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
        # If test_data_loaded_full_for_n_node remains empty, n_node is based on train only.
        pass
    else:
        try:
            with open(test_data_file, 'rb') as f:
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

    if n_node <= 1:
        print(f"Error: Invalid n_node ({n_node}). Must be > 1.")
        return

    # --- Data Subsetting (Applied after n_node and global graph source data is determined) ---
    if 0 < opt.data_subset_ratio < 1.0:
        train_sessions, train_targets = train_data_for_loader # Start with full train data
        num_original_train = len(train_sessions)
        if num_original_train > 0:
            num_train_to_keep = int(num_original_train * opt.data_subset_ratio)
            if num_train_to_keep == 0: num_train_to_keep = 1
            train_data_for_loader = (train_sessions[:num_train_to_keep], train_targets[:num_train_to_keep])
            print(f"--- Limiting training data to {len(train_data_for_loader[0])} samples ({opt.data_subset_ratio*100:.1f}%). ---")

    # --- Prepare Data for DataLoaders (train_set_for_data_obj, test_set_for_data_obj) ---
    train_set_for_data_obj = train_data_for_loader
    test_set_for_data_obj = ([], []) # Initialize for evaluation data

    if opt.validation:
        if not train_set_for_data_obj[0]: # Check if training data (possibly subsetted) is empty
            print("Error: Training data is empty, cannot perform validation split.")
            return
        print(f"Splitting training data for validation (portion: {opt.valid_portion}) from {len(train_set_for_data_obj[0])} samples.")
        train_set_for_data_obj, valid_set_for_data_obj = utils.split_validation(train_set_for_data_obj, opt.valid_portion)
        test_set_for_data_obj = valid_set_for_data_obj # Use validation set for evaluation
        print(f"Number of training samples after split: {len(train_set_for_data_obj[0])}")
        print(f"Number of validation samples: {len(test_set_for_data_obj[0])}")
    else:
        # Load actual test data for evaluation (not the one used for n_node if different)
        try:
            with open(test_data_file, 'rb') as f:
                test_set_for_data_obj = pickle.load(f)
            if not (isinstance(test_set_for_data_obj, tuple) and len(test_set_for_data_obj) == 2):
                print(f"Error: Test data at {test_data_file} is not in expected format. Using empty for eval.")
                test_set_for_data_obj = ([], [])
        except FileNotFoundError:
            print(f"Warning: Test data file for evaluation not found at {test_data_file}. Using empty for eval.")
            test_set_for_data_obj = ([], [])
        except Exception as e:
            print(f"Warning: Error loading test data for evaluation: {e}. Using empty for eval.")
        
        # Apply subsetting to test data if needed
        if 0 < opt.data_subset_ratio < 1.0 and test_set_for_data_obj[0]:
            test_sessions, test_targets = test_set_for_data_obj
            num_original_test = len(test_sessions)
            if num_original_test > 0:
                num_test_to_keep = int(num_original_test * opt.data_subset_ratio)
                if num_test_to_keep == 0: num_test_to_keep = 1
                test_set_for_data_obj = (test_sessions[:num_test_to_keep], test_targets[:num_test_to_keep])
                print(f"--- Limiting test data to {len(test_set_for_data_obj[0])} samples ({opt.data_subset_ratio*100:.1f}%). ---")
        print(f"Number of testing samples for evaluation: {len(test_set_for_data_obj[0])}")


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
            else: # Should not happen if build_graph_global_sparse returns a tensor
                 print("Error: Global graph construction returned None. Disabling global GCN.")
                 opt.global_gcn_layers = 0
        else:
            print("Warning: No source sessions to build global graph. Disabling global GCN.")
            opt.global_gcn_layers = 0
    else:
        print("Global GCN layers set to 0. Skipping global graph construction.")

    # --- Create Data Objects ---
    train_data_obj = None
    if train_set_for_data_obj and train_set_for_data_obj[0]:
        try:
            train_data_obj = utils.Data(train_set_for_data_obj, shuffle=True)
            print(f"Training Data object created with {train_data_obj.length} samples.")
        except ValueError as e:
            print(f"Error creating Training Data object: {e}. Cannot proceed with training.")
            return
    else:
        print("Training data is empty. Cannot proceed with training.")
        return

    eval_data_obj = None # For test/validation
    if test_set_for_data_obj and test_set_for_data_obj[0]:
        try:
            eval_data_obj = utils.Data(test_set_for_data_obj, shuffle=False)
            print(f"Evaluation Data object created with {eval_data_obj.length} samples.")
        except ValueError as e:
            print(f"Error creating Evaluation Data object: {e}. Evaluation will use empty data.")
            eval_data_obj = utils.Data(([], []), shuffle=False) # Fallback to empty
    else:
        print("Evaluation data is empty or not specified. Creating empty Evaluation Data object.")
        eval_data_obj = utils.Data(([], []), shuffle=False) # Fallback to empty

    # --- Initialize Model ---
    model = SessionGraph(opt, n_node, global_adj_sparse_matrix=global_adj_sparse_tensor)
    model.to(device)

    # --- Optimizer and Scheduler ---
    # These are now attributes of the model, as expected by train_test
    model.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    # --- Checkpoint and Training Loop ---
    checkpoint_dir = f'./checkpoints/{opt.dataset}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    start_epoch = 0
    best_result = [0.0, 0.0, 0.0] # Recall@K, MRR@K, Precision@K (Precision currently not primary)
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
            best_result = checkpoint.get('best_result', best_result)
            print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}.")
            print(f"Previous best result: Recall@{opt.k_metric}: {best_result[0]:.4f}, MRR@{opt.k_metric}: {best_result[1]:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
    else:
        if opt.resume_from_checkpoint:
            print(f"Checkpoint file not found: {opt.resume_from_checkpoint}. Starting fresh.")
        else:
            print("No checkpoint specified. Starting fresh.")

    start_time_training = time.time()

    for epoch_num in range(start_epoch, opt.epoch):
        current_lr = model.optimizer.param_groups[0]['lr']
        print(f"{'-'*25} Epoch: {epoch_num} | LR: {current_lr:.6f} {'-'*25}")

        if train_data_obj is None or train_data_obj.length == 0: # Should have been caught earlier
            print("Critical Error: train_data_obj is None or empty at training loop. Exiting.")
            return

        # train_test now performs one epoch of training and then evaluates
        recall, mrr, precision = train_test(model, train_data_obj, eval_data_obj, opt)
        
        flag = 0
        if eval_data_obj and eval_data_obj.length > 0: # Only consider improvement if eval data exists
            if recall >= best_result[0]: # Recall as primary metric
                best_result[0] = recall
                best_epoch[0] = epoch_num
                flag = 1 
            if mrr >= best_result[1]: # MRR as secondary metric
                best_result[1] = mrr
                best_epoch[1] = epoch_num
                if not flag: flag = 1 # Consider it an improvement if MRR improves even if Recall doesn't

            print(f'Current Best (k={opt.k_metric}): Recall: {best_result[0]:.4f} (Epoch {best_epoch[0]}), MRR: {best_result[1]:.4f} (Epoch {best_epoch[1]})')
            
            if flag == 1: # If Recall or MRR improved
                best_model_save_path = os.path.join(checkpoint_dir, f'model_best_k{opt.k_metric}.pth')
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'scheduler_state_dict': model.scheduler.state_dict(),
                    'best_result': best_result,
                    'opt': opt
                }, best_model_save_path)
                print(f'Best model checkpoint saved to {best_model_save_path}')
                bad_counter = 0 # Reset bad_counter on improvement
            else:
                bad_counter += 1
            
            if bad_counter >= opt.patience:
                print(f"Early stopping after {opt.patience} epochs without improvement.")
                break
        else: # No evaluation data
            print(f'Epoch {epoch_num} completed (no evaluation data).')


    total_training_time = time.time() - start_time_training
    print(f"{'-'*25} Training Finished {'-'*25}")
    print(f"Total Training Run time: {total_training_time:.2f} s")

    if eval_data_obj and eval_data_obj.length > 0:
        print(f"Final Best Overall Result (k={opt.k_metric}):")
        print(f'\tRecall@{opt.k_metric}: {best_result[0]:.4f} (Achieved at Epoch {best_epoch[0]})')
        print(f'\tMRR@{opt.k_metric}: {best_result[1]:.4f} (Achieved at Epoch {best_epoch[1]})')
    else:
        print("Training finished. No evaluation data was available to determine best results.")

if __name__ == '__main__':
    main()