# utils.py

import numpy as np
import torch
# import networkx as nx # اگر برای ساخت گراف کلی استفاده می‌شد، لازم بود

def normalize_adj_symmetric(adj):
    """Symmetrically normalize adjacency matrix."""
    # افزودن یال به خود (self-loops)
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def build_graph_global(all_sessions_items, n_node):
    """
    Builds a global adjacency matrix from all session items and normalizes it.
    Returns a single normalized adjacency matrix (sparse if possible in future).
    """
    if not all_sessions_items:
        print("Warning: Trying to build global graph from empty session list.")
        adj = np.zeros((n_node, n_node), dtype=np.float32)
        # برای گراف خالی هم یال به خود اضافه می‌کنیم تا نرمال‌سازی خطا ندهد
        np.fill_diagonal(adj, 1) # یا حداقل یک یال به خود برای نود 0 اگر n_node > 0
        return normalize_adj_symmetric(adj)


    # ساخت ماتریس همسایگی خام (فقط ارتباطات بین آیتم‌های مختلف در یک سشن)
    # در اینجا از یک ماتریس همسایگی ساده استفاده می‌کنیم که اگر آیتم i و j پشت سر هم آمده باشند، یک یال داریم
    # برای سادگی، جهت را در نظر نمی‌گیریم و یک گراف غیر جهت‌دار می‌سازیم.
    # TAGNN ممکن است از روش پیچیده‌تری برای ساخت گراف گلوبال استفاده کند (مثلا مبتنی بر هم‌رخدادی کلی).
    # این یک پیاده‌سازی پایه است.
    adj_raw = np.zeros((n_node, n_node), dtype=np.float32)

    for session_items in all_sessions_items:
        cleaned_session = []
        for item_wrapper in session_items:
            item_val = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
            if isinstance(item_val, int) and 0 < item_val < n_node: # آیتم 0 پدینگ است
                cleaned_session.append(item_val)

        for i in range(len(cleaned_session) - 1):
            u, v = cleaned_session[i], cleaned_session[i+1]
            # برای گراف غیر جهت‌دار
            adj_raw[u, v] = 1.0
            adj_raw[v, u] = 1.0 # اگر می‌خواهید گراف جهت‌دار باشد، این خط را حذف کنید

    print(f"Global graph: Found {np.sum(adj_raw > 0) / 2:.0f} unique edges from sessions.")
    
    # نرمال‌سازی ماتریس همسایگی
    # توجه: اگر n_node بسیار بزرگ باشد، این ماتریس چگال می‌تواند مشکل‌ساز شود.
    # در آن صورت باید از ماتریس‌های پراکنده (sparse) استفاده کرد.
    normalized_adj = normalize_adj_symmetric(adj_raw)
    
    # در این پیاده‌سازی ساده، adj_in و adj_out گلوبال یکی هستند (ماتریس نرمال شده غیر جهت‌دار)
    # اگر نیاز به adj_in و adj_out جداگانه برای گراف گلوبال دارید، باید منطق ساخت و نرمال‌سازی را گسترش دهید.
    return normalized_adj


def data_masks(all_usr_pois, item_tail=0):
    """
    Pads sequences and creates masks.
    all_usr_pois: list of sessions, where each session is a list of item IDs.
    item_tail: value used for padding (usually 0).
    """
    if not all_usr_pois:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.bool), 0

    us_lens = [len(upois) for upois in all_usr_pois if upois]
    if not us_lens:
        max_len = 0
    else:
        max_len = max(us_lens)

    if max_len == 0:
        num_sessions = len(all_usr_pois)
        return torch.zeros((num_sessions, 0), dtype=torch.long), torch.zeros((num_sessions, 0), dtype=torch.bool), 0

    us_pois_padded = []
    us_msks = []

    for upois_original in all_usr_pois:
        cleaned_upois = []
        if upois_original:
            for item_wrapper in upois_original:
                item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
                if isinstance(item, (int, float)) and not np.isnan(item):
                    cleaned_upois.append(int(item))

        padding_len = max_len - len(cleaned_upois)
        padded_seq = cleaned_upois + [item_tail] * padding_len
        us_pois_padded.append(padded_seq)

        mask = [True] * len(cleaned_upois) + [False] * padding_len
        us_msks.append(mask)

    try:
        us_pois_tensor = torch.tensor(us_pois_padded, dtype=torch.long)
        us_msks_tensor = torch.tensor(us_msks, dtype=torch.bool)
    except Exception as e:
        print("Error in tensor conversion (data_masks):")
        print(f"Max length: {max_len}")
        print(f"Number of sequences to pad: {len(all_usr_pois)}")
        raise e

    return us_pois_tensor, us_msks_tensor, max_len


def split_validation(train_set_original, valid_portion):
    if not (isinstance(train_set_original, tuple) and len(train_set_original) == 2):
        raise ValueError("train_set for split_validation must be a tuple of (sessions, targets)")

    train_set_x, train_set_y = train_set_original

    if not train_set_x:
        print("Warning: train_set_x is empty in split_validation. Returning empty sets.")
        return ([], []), ([], [])

    n_samples = len(train_set_x)
    if n_samples == 0:
        return ([], []), ([], [])

    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)

    n_train = int(np.round(n_samples * (1. - valid_portion)))
    n_train = max(0, min(n_samples, n_train))

    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x_split = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y_split = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x_split, train_set_y_split), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data_tuple, shuffle=False, graph=None):
         if not isinstance(data_tuple, tuple) or len(data_tuple) != 2:
              raise ValueError("Input data for Data class must be a tuple of (sessions, targets)")

         inputs_raw, targets_raw = data_tuple

         if len(inputs_raw) != len(targets_raw):
              raise ValueError(f"Number of sessions ({len(inputs_raw)}) does not match number of targets ({len(targets_raw)})")

         cleaned_targets_raw = []
         for t_wrapper in targets_raw:
             t = t_wrapper[0] if isinstance(t_wrapper, list) and t_wrapper else t_wrapper
             if isinstance(t, (int, float)) and not np.isnan(t):
                 cleaned_targets_raw.append(int(t))
             else:
                 cleaned_targets_raw.append(0)

         inputs_tensor, mask_tensor, len_max_seq = data_masks(inputs_raw, item_tail=0)

         self.inputs = inputs_tensor.numpy()
         self.mask = mask_tensor.float().numpy()
         self.len_max = len_max_seq
         self.targets = np.asarray(cleaned_targets_raw, dtype=np.int64)
         self.length = len(inputs_raw)
         self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.length == 0: return []

        indices = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(indices)

        n_batch = self.length // batch_size
        if self.length % batch_size != 0:
            n_batch += 1

        slices = []
        for i in range(n_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.length)
            if start < end:
                slices.append(indices[start:end])
        return slices


    def get_slice(self, i_indices):
        inputs_original_seqs_batch = self.inputs[i_indices]
        mask_slice_batch = self.mask[i_indices]
        targets_slice_batch = self.targets[i_indices]

        current_batch_size = len(inputs_original_seqs_batch)
        current_max_seq_len = self.len_max

        session_unique_nodes_list = []
        max_n_node_in_batch = 0

        for k in range(current_batch_size):
            actual_seq_len = int(mask_slice_batch[k].sum())
            original_seq_k = inputs_original_seqs_batch[k][:actual_seq_len]
            unique_nodes_in_seq_k = np.unique(original_seq_k[original_seq_k > 0])
            session_unique_nodes_list.append(unique_nodes_in_seq_k)
            max_n_node_in_batch = max(max_n_node_in_batch, len(unique_nodes_in_seq_k))

        if max_n_node_in_batch == 0:
             max_n_node_in_batch = 1

        items_for_gnn_batch = np.zeros((current_batch_size, max_n_node_in_batch), dtype=np.int64)
        alias_inputs_for_transformer_batch = np.zeros((current_batch_size, current_max_seq_len), dtype=np.int64)
        A_batch_list = []

        for k in range(current_batch_size):
            unique_nodes_k = session_unique_nodes_list[k]
            num_unique_in_session_k = len(unique_nodes_k)

            if num_unique_in_session_k > 0:
                items_for_gnn_batch[k, :num_unique_in_session_k] = unique_nodes_k

            node_map_session_k = {node_id: local_idx for local_idx, node_id in enumerate(unique_nodes_k)}

            original_seq_padded_k = inputs_original_seqs_batch[k]
            for j, item_id_original in enumerate(original_seq_padded_k[:current_max_seq_len]):
                alias_inputs_for_transformer_batch[k, j] = node_map_session_k.get(item_id_original, 0)

            adj_session_k_dense = np.zeros((num_unique_in_session_k, num_unique_in_session_k), dtype=np.float32)
            actual_seq_len_k = int(mask_slice_batch[k].sum())
            current_seq_local_indices_k = alias_inputs_for_transformer_batch[k, :actual_seq_len_k]

            for j in range(actual_seq_len_k - 1):
                u_local_idx = current_seq_local_indices_k[j]
                v_local_idx = current_seq_local_indices_k[j+1]
                original_u = inputs_original_seqs_batch[k, j]
                original_v = inputs_original_seqs_batch[k, j+1]

                if original_u > 0 and original_v > 0:
                    if u_local_idx < num_unique_in_session_k and v_local_idx < num_unique_in_session_k:
                         adj_session_k_dense[u_local_idx, v_local_idx] = 1.0

            sum_in = np.sum(adj_session_k_dense, axis=0, keepdims=True)
            sum_in[sum_in == 0] = 1
            adj_in_norm_k = adj_session_k_dense / sum_in

            sum_out = np.sum(adj_session_k_dense, axis=1, keepdims=True)
            sum_out[sum_out == 0] = 1
            adj_out_norm_k = adj_session_k_dense / sum_out

            padded_A_in_k = np.zeros((max_n_node_in_batch, max_n_node_in_batch), dtype=np.float32)
            padded_A_out_k = np.zeros((max_n_node_in_batch, max_n_node_in_batch), dtype=np.float32)

            if num_unique_in_session_k > 0:
                padded_A_in_k[:num_unique_in_session_k, :num_unique_in_session_k] = adj_in_norm_k
                padded_A_out_k[:num_unique_in_session_k, :num_unique_in_session_k] = adj_out_norm_k

            A_combined_k = np.concatenate([padded_A_in_k, padded_A_out_k], axis=1)
            A_batch_list.append(A_combined_k)

        A_batch_tensor = np.array(A_batch_list, dtype=np.float32)

        return (
            alias_inputs_for_transformer_batch,
            A_batch_tensor,
            items_for_gnn_batch,
            mask_slice_batch,
            targets_slice_batch
        )