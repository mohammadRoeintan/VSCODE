import networkx as nx
import numpy as np
import torch
# import scipy sparse if needed
# from scipy.sparse import lil_matrix, csr_matrix # Not used in current build_graph

# تابع ساخت گراف (بدون تغییر)
def build_graph(sessions):
    """Builds adjacency matrices from session data."""
    if not sessions:
        raise ValueError("Cannot build graph from empty session list.")

    max_node = 0
    for seq in sessions:
        if seq:
             try:
                  max_node = max(max_node, max(item for item in seq if isinstance(item, int) and item > 0))
             except ValueError:
                  pass

    if max_node == 0:
        print("Warning: No valid items found in sessions. Returning minimal graph.")
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    graph_size = max_node + 1
    adj_in_counts = {i: {} for i in range(graph_size)}
    adj_out_counts = {i: {} for i in range(graph_size)}

    for seq in sessions:
        for i in range(len(seq) - 1):
             u, v = seq[i], seq[i+1]
             if isinstance(u, int) and isinstance(v, int) and u > 0 and v > 0 and u < graph_size and v < graph_size:
                 adj_out_counts[u][v] = adj_out_counts[u].get(v, 0) + 1
                 adj_in_counts[v][u] = adj_in_counts[v].get(u, 0) + 1

    adj_in = np.zeros((graph_size, graph_size), dtype=np.float32)
    adj_out = np.zeros((graph_size, graph_size), dtype=np.float32)

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


# تابع data_masks (بدون تغییر نسبت به نسخه قبلی)
def data_masks(all_usr_pois, item_tail=0):
    if not all_usr_pois:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), 0

    us_lens = [len(upois) for upois in all_usr_pois]
    if not us_lens: max_len = 0
    else: max_len = max(us_lens) if us_lens else 0

    if max_len == 0:
         num_sessions = len(all_usr_pois)
         return torch.zeros((num_sessions, 0), dtype=torch.long), torch.zeros((num_sessions, 0), dtype=torch.float32), 0

    us_pois_padded = []
    us_msks = []
    for upois in all_usr_pois:
        cleaned_upois = [int(item) for item in upois if isinstance(item, (int, float)) and item is not None]
        padding_len = max_len - len(cleaned_upois)
        padded_seq = cleaned_upois + [item_tail] * padding_len
        us_pois_padded.append(padded_seq)
        mask = [1] * len(cleaned_upois) + [0] * padding_len
        us_msks.append(mask)

    try:
        us_pois_tensor = torch.tensor(us_pois_padded, dtype=torch.long)
        us_msks_tensor = torch.tensor(us_msks, dtype=torch.float32)
    except Exception as e:
        print("Error in tensor conversion (data_masks):")
        print(f"Max length: {max_len}")
        print(f"Number of sequences: {len(us_pois_padded)}")
        for i, seq in enumerate(us_pois_padded):
             if len(seq) != max_len:
                  print(f"Inconsistent length at index {i}: expected {max_len}, got {len(seq)}")
        raise e
    return us_pois_tensor, us_msks_tensor, max_len


# تابع split_validation (بدون تغییر)
def split_validation(train_set, valid_portion):
    if not isinstance(train_set, tuple) or len(train_set) != 2:
        raise ValueError("train_set must be a tuple of (sessions, targets)")
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    if n_samples == 0: return ([], []), ([], [])
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    n_train = max(0, min(n_samples, n_train))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


# کلاس Data (بدون تغییر در __init__ و generate_batch نسبت به نسخه قبلی)
class Data():
    def __init__(self, data, shuffle=False, graph=None):
         if not isinstance(data, tuple) or len(data) != 2:
              raise ValueError("Input data must be a tuple of (sessions, targets)")
         inputs_raw, targets_raw = data
         if len(inputs_raw) != len(targets_raw):
              raise ValueError(f"Number of sessions ({len(inputs_raw)}) does not match number of targets ({len(targets_raw)})")

         inputs_tensor, mask_tensor, len_max = data_masks(inputs_raw, item_tail=0)
         self.inputs = inputs_tensor.numpy()
         self.mask = mask_tensor.numpy()
         self.len_max = len_max
         self.targets = np.asarray(targets_raw)
         self.length = len(inputs_raw)
         self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.length == 0: return []
        indices = np.arange(self.length)
        if self.shuffle: np.random.shuffle(indices)
        n_batch = self.length // batch_size
        if self.length % batch_size != 0: n_batch += 1
        slices = []
        for i in range(n_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.length)
            slices.append(indices[start:end])
        slices = [s for s in slices if len(s) > 0]
        return slices


    # ***** تابع get_slice (اصلاح شده اساسی) *****
    def get_slice(self, i):
        """Gets a slice of data based on indices i and prepares inputs for GNN."""
        # i is an array of indices from generate_batch
        inputs_original_seqs = self.inputs[i] # Original sequences with padding (Batch, SeqLen)
        mask_slice = self.mask[i]             # Mask for original sequences (Batch, SeqLen)
        targets_slice = self.targets[i]       # Targets for sequences (Batch,)
        batch_size = len(inputs_original_seqs)

        # 1. Find unique nodes *per session* and the overall max number of unique nodes in batch
        session_unique_nodes_list = []
        max_n_node_in_batch = 0
        for k in range(batch_size):
            # Get actual items in the sequence using the mask
            seq_len = int(mask_slice[k].sum())
            original_seq = inputs_original_seqs[k][:seq_len]
            # Find unique non-padding nodes in this sequence
            unique_nodes = np.unique(original_seq[original_seq > 0]) # Exclude padding 0
            session_unique_nodes_list.append(unique_nodes)
            max_n_node_in_batch = max(max_n_node_in_batch, len(unique_nodes))

        # Handle case where batch might only contain empty sequences or padding
        if max_n_node_in_batch == 0:
             max_n_node_in_batch = 1 # Need at least one node dimension for tensors

        # 2. Create the 2D 'items' tensor and the node mapping for each session
        # 'items' will hold the original node IDs for embedding lookup
        items_batch = np.zeros((batch_size, max_n_node_in_batch), dtype=np.int64)
        # 'alias_inputs' will hold sequences mapped to indices within each row of 'items_batch'
        alias_inputs_batch = np.zeros_like(inputs_original_seqs, dtype=np.int64)
        # List to hold adjacency matrices A for each session
        A_batch = []

        for k in range(batch_size):
            unique_nodes = session_unique_nodes_list[k]
            num_unique_in_session = len(unique_nodes)

            # Create the k-th row of 'items' (original IDs, padded)
            if num_unique_in_session > 0:
                items_batch[k, :num_unique_in_session] = unique_nodes
            # items_batch[k, num_unique_in_session:] = 0 # Already initialized to 0

            # Create mapping from original node ID -> index within this session's unique nodes (0 to num_unique_in_session-1)
            node_map_session = {node_id: idx for idx, node_id in enumerate(unique_nodes)}

            # Create the k-th row of 'alias_inputs' using the session map
            original_seq_padded = inputs_original_seqs[k]
            for j, item_id in enumerate(original_seq_padded):
                alias_inputs_batch[k, j] = node_map_session.get(item_id, 0) # Map to local index, default to 0 (padding) if item_id not in unique_nodes (e.g., was padding)

            # --- Build Adjacency Matrix (A) for session k ---
            # Use local indices (0 to num_unique_in_session-1)
            adj_session = np.zeros((num_unique_in_session, num_unique_in_session), dtype=np.float32)
            seq_len = int(mask_slice[k].sum())
            current_seq_local_indices = alias_inputs_batch[k, :seq_len]

            for j in range(seq_len - 1):
                u_local_idx = current_seq_local_indices[j]
                v_local_idx = current_seq_local_indices[j+1]
                # Ensure indices are valid (should be, as they come from mapping unique nodes)
                if u_local_idx < num_unique_in_session and v_local_idx < num_unique_in_session:
                     # We only care about transitions between non-padding nodes mapped locally
                     # The map defaults padding to 0, but check original IDs were > 0?
                     # Let's assume the alias map handles this correctly.
                     # If u_local_idx and v_local_idx correspond to original padding (0), they won't be in node_map_session keys.
                     # We should only add edge if both u and v were non-padding originally.
                     original_u = inputs_original_seqs[k, j]
                     original_v = inputs_original_seqs[k, j+1]
                     if original_u > 0 and original_v > 0:
                          adj_session[u_local_idx, v_local_idx] = 1


            # Normalize A (In/Out degrees) for the session
            sum_in = np.sum(adj_session, 0)
            sum_in[sum_in == 0] = 1
            adj_in_norm = adj_session / sum_in # Broadcasting

            sum_out = np.sum(adj_session, 1)
            sum_out[sum_out == 0] = 1
            # Need to transpose correctly for division by sum_out column vector
            adj_out_norm = (adj_session.T / sum_out).T

            # Pad adjacency matrices to batch max size (max_n_node_in_batch)
            padded_A_in = np.zeros((max_n_node_in_batch, max_n_node_in_batch), dtype=np.float32)
            padded_A_out = np.zeros((max_n_node_in_batch, max_n_node_in_batch), dtype=np.float32)

            if num_unique_in_session > 0:
                padded_A_in[:num_unique_in_session, :num_unique_in_session] = adj_in_norm
                padded_A_out[:num_unique_in_session, :num_unique_in_session] = adj_out_norm

            # Concatenate normalized matrices for this session
            A_combined = np.concatenate([padded_A_in, padded_A_out], axis=1) # Shape: (max_n_node, 2 * max_n_node)
            A_batch.append(A_combined)


        # Stack all session A matrices into a single 3D tensor for the batch
        A_batch_tensor = np.array(A_batch) # Shape: (batch_size, max_n_node, 2 * max_n_node)

        # Return the processed batch data
        return (
            alias_inputs_batch,  # Sequences mapped to indices within items_batch rows (Batch, SeqLen)
            A_batch_tensor,      # Batched adjacency matrices (Batch, max_n_node, 2*max_n_node)
            items_batch,         # Original node IDs for embedding (Batch, max_n_node)
            mask_slice,          # Original sequence mask (Batch, SeqLen)
            targets_slice        # Original targets (Batch,)
        )