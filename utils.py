import numpy as np
import torch
# import networkx as nx # اگر برای ساخت گراف کلی استفاده می‌شد، لازم بود

# تابع build_graph_global (بدون تغییر، همچنان استفاده نمی‌شود)
def build_graph_global(all_sessions_items):
    if not all_sessions_items:
        print("Warning: Trying to build global graph from empty session list.")
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    unique_nodes = set()
    for session in all_sessions_items:
        for item_wrapper in session:
            item = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
            if isinstance(item, int) and item > 0:
                unique_nodes.add(item)

    if not unique_nodes:
        print("Warning: No valid nodes found to build global graph.")
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    n_node = max(unique_nodes) + 1

    adj_in_counts = {i: {} for i in range(n_node)}
    adj_out_counts = {i: {} for i in range(n_node)}

    for session_items in all_sessions_items:
        cleaned_session = []
        for item_wrapper in session_items:
            item_val = item_wrapper[0] if isinstance(item_wrapper, list) and item_wrapper else item_wrapper
            if isinstance(item_val, int) and item_val > 0 and item_val < n_node:
                cleaned_session.append(item_val)

        for i in range(len(cleaned_session) - 1):
            u, v = cleaned_session[i], cleaned_session[i+1]
            adj_out_counts[u][v] = adj_out_counts[u].get(v, 0) + 1
            adj_in_counts[v][u] = adj_in_counts[v].get(u, 0) + 1

    adj_in = np.zeros((n_node, n_node), dtype=np.float32)
    adj_out = np.zeros((n_node, n_node), dtype=np.float32)

    for u, neighbors in adj_out_counts.items():
        out_degree = sum(neighbors.values())
        if out_degree > 0:
            for v, count in neighbors.items():
                adj_out[u, v] = count / out_degree

    for v, neighbors in adj_in_counts.items():
        in_degree = sum(neighbors.values())
        if in_degree > 0:
            for u, count in neighbors.items():
                adj_in[v, u] = count / in_degree
    return adj_in, adj_out


def data_masks(all_usr_pois, item_tail=0):
    if not all_usr_pois:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.float32), 0

    us_lens = [len(upois) for upois in all_usr_pois if upois]
    if not us_lens:
        max_len = 0
    else:
        max_len = max(us_lens)

    if max_len == 0:
        num_sessions = len(all_usr_pois)
        # dtype=torch.bool می‌تواند برای ماسک‌ها کمی بهینه‌تر باشد اگر فقط 0 و 1 هستند
        return torch.zeros((num_sessions, 0), dtype=torch.long), torch.zeros((num_sessions, 0), dtype=torch.bool), 0

    us_pois_padded = []
    us_msks = [] # برای ماسک از نوع bool استفاده می‌کنیم

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

        # ایجاد ماسک به صورت bool
        mask = [True] * len(cleaned_upois) + [False] * padding_len
        us_msks.append(mask)

    try:
        # ایجاد تنسورها مستقیماً با نوع داده نهایی
        us_pois_tensor = torch.tensor(us_pois_padded, dtype=torch.long)
        us_msks_tensor = torch.tensor(us_msks, dtype=torch.bool) # استفاده از bool برای ماسک
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

         # data_masks حالا تنسورهای PyTorch برمی‌گرداند
         inputs_tensor, mask_tensor, len_max_seq = data_masks(inputs_raw, item_tail=0)

         # به جای تبدیل به NumPy و سپس به تنسور در model.py، مستقیماً تنسورها را نگه می‌داریم
         # یا اگر می‌خواهید همچنان NumPy باشد برای سازگاری، تغییر ندهید.
         # برای بهینگی حداکثری در انتقال به GPU، بهتر است از ابتدا تنسور باشند.
         # اما چون get_slice همچنان با NumPy کار می‌کند، این بخش را NumPy نگه می‌داریم.
         self.inputs = inputs_tensor.numpy() # (num_samples, max_len)
         self.mask = mask_tensor.numpy(dtype=np.float32) # تبدیل bool به float32 برای سازگاری با کد قبلی mask
                                                          # یا می‌توانید mask_tensor.float().numpy() استفاده کنید
                                                          # یا در model.py نوع ماسک را مدیریت کنید (long یا bool)
         self.len_max = len_max_seq
         self.targets = np.asarray(cleaned_targets_raw, dtype=np.int64) # dtype=np.int64 برای سازگاری با long در PyTorch
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
        mask_slice_batch = self.mask[i_indices] # حالا float32 است
        targets_slice_batch = self.targets[i_indices]

        current_batch_size = len(inputs_original_seqs_batch)
        # از self.len_max که قبلا محاسبه شده استفاده می‌کنیم
        current_max_seq_len_in_batch = self.len_max # یا می‌توانید برای هر بچ جداگانه محاسبه کنید

        session_unique_nodes_list = []
        max_n_node_in_batch = 0

        # این حلقه همچنان با NumPy کار می‌کند.
        # بهینه‌سازی این بخش برای اجرا روی GPU بسیار پیچیده است و نیاز به بازنویسی کامل دارد.
        for k in range(current_batch_size):
            actual_seq_len = int(mask_slice_batch[k].sum()) # mask_slice_batch حالا float است
            original_seq_k = inputs_original_seqs_batch[k][:actual_seq_len]
            unique_nodes_in_seq_k = np.unique(original_seq_k[original_seq_k > 0])
            session_unique_nodes_list.append(unique_nodes_in_seq_k)
            max_n_node_in_batch = max(max_n_node_in_batch, len(unique_nodes_in_seq_k))

        if max_n_node_in_batch == 0:
             max_n_node_in_batch = 1

        # این آرایه‌ها همچنان NumPy هستند و در model.py به تنسور GPU تبدیل می‌شوند
        items_for_gnn_batch = np.zeros((current_batch_size, max_n_node_in_batch), dtype=np.int64)
        alias_inputs_for_transformer_batch = np.zeros((current_batch_size, current_max_seq_len_in_batch), dtype=np.int64)
        A_batch_list = [] # این لیست همچنان آرایه‌های NumPy نگه می‌دارد

        for k in range(current_batch_size):
            unique_nodes_k = session_unique_nodes_list[k]
            num_unique_in_session_k = len(unique_nodes_k)

            if num_unique_in_session_k > 0:
                items_for_gnn_batch[k, :num_unique_in_session_k] = unique_nodes_k

            node_map_session_k = {node_id: local_idx for local_idx, node_id in enumerate(unique_nodes_k)}

            original_seq_padded_k = inputs_original_seqs_batch[k]
            # به جای حلقه، از نگاشت برداری شده استفاده می‌کنیم (اگرچه node_map داینامیک است)
            # این بخش همچنان نیاز به حلقه دارد چون node_map برای هر k متفاوت است
            for j, item_id_original in enumerate(original_seq_padded_k):
                # فقط تا طول واقعی توالی این بچ پردازش می‌کنیم
                if j < current_max_seq_len_in_batch : # اطمینان از عدم خروج از محدوده alias_inputs
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
                         adj_session_k_dense[u_local_idx, v_local_idx] = 1.0 # اطمینان از float بودن

            sum_in = np.sum(adj_session_k_dense, axis=0, keepdims=True)
            sum_in[sum_in == 0] = 1
            adj_in_norm_k = adj_session_k_dense / sum_in

            sum_out = np.sum(adj_session_k_dense, axis=1, keepdims=True)
            sum_out[sum_out == 0] = 1
            adj_out_norm_k = adj_session_k_dense / sum_out # تقسیم ردیفی با broadcasting

            padded_A_in_k = np.zeros((max_n_node_in_batch, max_n_node_in_batch), dtype=np.float32)
            padded_A_out_k = np.zeros((max_n_node_in_batch, max_n_node_in_batch), dtype=np.float32)

            if num_unique_in_session_k > 0:
                padded_A_in_k[:num_unique_in_session_k, :num_unique_in_session_k] = adj_in_norm_k
                padded_A_out_k[:num_unique_in_session_k, :num_unique_in_session_k] = adj_out_norm_k

            A_combined_k = np.concatenate([padded_A_in_k, padded_A_out_k], axis=1)
            A_batch_list.append(A_combined_k)

        A_batch_tensor = np.array(A_batch_list, dtype=np.float32) # اطمینان از dtype نهایی

        return (
            alias_inputs_for_transformer_batch, # (B, self.len_max یا current_max_seq_len)
            A_batch_tensor,
            items_for_gnn_batch,
            mask_slice_batch, # (B, self.len_max) - float32
            targets_slice_batch
        )