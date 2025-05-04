import networkx as nx
import numpy as np
import torch
# import scipy sparse if needed, but it seems build_graph doesn't use it anymore
# from scipy.sparse import lil_matrix, csr_matrix

# تابع ساخت گراف (بدون تغییر منطق اصلی، فقط بررسی داده ورودی)
def build_graph(sessions):
    """Builds adjacency matrices from session data."""
    if not sessions:
        print("Warning: build_graph called with empty sessions list.")
        # Return empty adjacency matrices or handle as appropriate
        # For now, let's return None or raise error? Let's raise for clarity.
        raise ValueError("Cannot build graph from empty session list.")

    max_node = 0
    for seq in sessions:
        if seq: # Check if sequence is not empty
             # Handle potential non-integer items gracefully
             try:
                  max_node = max(max_node, max(item for item in seq if isinstance(item, int) and item > 0))
             except ValueError: # Handles cases where seq might be empty after filtering
                  pass # Keep previous max_node

    if max_node == 0:
         print("Warning: No valid items found in sessions to determine graph size. Max node set to 0.")
         # Decide how to handle this - perhaps a default size or error
         # Let's assume node indices start from 1, so max_node should be at least 1 if data exists
         # But if all items are 0 (padding), this might happen.
         # Let's return small empty matrices?
         return np.zeros((1, 1)), np.zeros((1, 1)) # Return minimal structure

    # +1 because node indices might be 0-based or 1-based, ensure size accommodates max_node index
    graph_size = max_node + 1
    # Using dictionaries for sparse representation initially might be more memory efficient
    adj_in_counts = {i: {} for i in range(graph_size)}
    adj_out_counts = {i: {} for i in range(graph_size)}

    for seq in sessions:
        for i in range(len(seq) - 1):
             # Ensure items are valid integers and within expected range
             u = seq[i]
             v = seq[i+1]
             if isinstance(u, int) and isinstance(v, int) and u > 0 and v > 0 and u < graph_size and v < graph_size:
                 # Increment edge counts
                 adj_out_counts[u][v] = adj_out_counts[u].get(v, 0) + 1
                 adj_in_counts[v][u] = adj_in_counts[v].get(u, 0) + 1

    # Convert counts to normalized adjacency matrices (numpy arrays for now)
    # This part is memory intensive for large graphs if dense
    adj_in = np.zeros((graph_size, graph_size), dtype=np.float32)
    adj_out = np.zeros((graph_size, graph_size), dtype=np.float32)

    for u, neighbors in adj_out_counts.items():
        out_degree = sum(neighbors.values())
        if out_degree > 0:
            for v, count in neighbors.items():
                adj_out[u, v] = count / out_degree # Normalize by out-degree

    for v, neighbors in adj_in_counts.items():
        in_degree = sum(neighbors.values())
        if in_degree > 0:
            for u, count in neighbors.items():
                adj_in[v, u] = count / in_degree # Normalize by in-degree


    # Returning numpy arrays as the GNN expects tensors created from numpy arrays later
    return adj_in, adj_out


# تابع data_masks (اصلاح شده برای مدیریت بهتر padding و انواع داده)
def data_masks(all_usr_pois, item_tail=0): # استفاده از item_tail عددی
    """Pads sequences and creates masks."""
    # Handle potential empty input
    if not all_usr_pois:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), 0

    us_lens = [len(upois) for upois in all_usr_pois]
    # Handle case where all sequences might be empty
    if not us_lens:
         max_len = 0
    else:
         max_len = max(us_lens) if us_lens else 0


    # اگر همه دنباله‌ها خالی باشند
    if max_len == 0:
         # Return empty tensors with appropriate dimensions if possible
         num_sessions = len(all_usr_pois)
         return torch.zeros((num_sessions, 0), dtype=torch.long), torch.zeros((num_sessions, 0), dtype=torch.long), 0


    us_pois_padded = []
    us_msks = []

    for upois in all_usr_pois:
        # Clean sequence: ensure items are integers, handle None or other types
        cleaned_upois = [int(item) for item in upois if isinstance(item, (int, float)) and item is not None] # Allow float conversion

        # Pad sequence
        padding_len = max_len - len(cleaned_upois)
        padded_seq = cleaned_upois + [item_tail] * padding_len
        us_pois_padded.append(padded_seq)

        # Create mask
        mask = [1] * len(cleaned_upois) + [0] * padding_len
        us_msks.append(mask)

    # تبدیل به تانسور
    try:
        # Ensure all sequences have the same length 'max_len' before creating tensor
        us_pois_tensor = torch.tensor(us_pois_padded, dtype=torch.long)
        us_msks_tensor = torch.tensor(us_msks, dtype=torch.float32) # Mask often used as float
    except Exception as e:
        print("Error in tensor conversion (data_masks):")
        # print("Sample us_pois_padded:", us_pois_padded[:2])
        # print("Sample us_msks:", us_msks[:2])
        print(f"Maximum length: {max_len}")
        print(f"Number of sequences: {len(us_pois_padded)}")
        # Check lengths consistency
        for i, seq in enumerate(us_pois_padded):
             if len(seq) != max_len:
                  print(f"Inconsistent length at index {i}: expected {max_len}, got {len(seq)}")
        raise e

    return us_pois_tensor, us_msks_tensor, max_len


def split_validation(train_set, valid_portion):
    # Ensure train_set is a tuple (sessions, targets)
    if not isinstance(train_set, tuple) or len(train_set) != 2:
        raise ValueError("train_set must be a tuple of (sessions, targets)")

    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    if n_samples == 0:
        # Handle empty training set
        return ([], []), ([], [])

    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))

    # Handle edge case where n_train might be 0 or n_samples
    n_train = max(0, min(n_samples, n_train))

    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
         # Ensure data is a tuple (sessions, targets)
         if not isinstance(data, tuple) or len(data) != 2:
              raise ValueError("Input data must be a tuple of (sessions, targets)")
         inputs_raw, targets_raw = data

         # Basic validation
         if len(inputs_raw) != len(targets_raw):
              raise ValueError(f"Number of sessions ({len(inputs_raw)}) does not match number of targets ({len(targets_raw)})")


         # Use item_tail=0 consistent with embedding padding_idx
         inputs_tensor, mask_tensor, len_max = data_masks(inputs_raw, item_tail=0)
         self.inputs = inputs_tensor.numpy() # Store as numpy for slicing compatibility
         self.mask = mask_tensor.numpy()
         self.len_max = len_max
         # Ensure targets are numpy array
         self.targets = np.asarray(targets_raw)

         self.length = len(inputs_raw)
         self.shuffle = shuffle
         # graph is not used in this class structure, graph construction happens outside
         # self.graph = graph # Commented out as it's not used

    def generate_batch(self, batch_size):
        """Generates batch indices."""
        if self.length == 0:
             return [] # No batches if no data

        indices = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(indices)

        # Adjust slices to avoid empty last batch if length % batch_size == 0
        n_batch = self.length // batch_size
        if self.length % batch_size != 0:
            n_batch += 1

        slices = []
        for i in range(n_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.length)
            slices.append(indices[start:end]) # Append slices of indices

        # Filter out potential empty slices just in case
        slices = [s for s in slices if len(s) > 0]

        return slices # Return list of index arrays


    def get_slice(self, i):
        """Gets a slice of data based on indices i."""
        # i is now an array of indices from generate_batch
        inputs_slice = self.inputs[i]
        mask_slice = self.mask[i]
        targets_slice = self.targets[i]

        items, n_node, A, alias_inputs = [], [], [], []
        # Find unique nodes *within the batch slice* to determine max_n_node for this batch
        # This makes graph construction dynamic per batch - more efficient?

        # 1. Find all unique nodes in the batch and max node ID
        batch_nodes = set([0]) # Include 0 for padding node
        max_node_in_batch = 0
        for u_input in inputs_slice:
             valid_nodes = set(item for item in u_input if item > 0) # Exclude padding
             batch_nodes.update(valid_nodes)
             if valid_nodes:
                  max_node_in_batch = max(max_node_in_batch, max(valid_nodes))


        # Map original node IDs to contiguous indices within the batch [0, num_unique_nodes-1]
        # Keep 0 mapped to 0
        unique_nodes_list = sorted(list(batch_nodes))
        node_map = {node_id: idx for idx, node_id in enumerate(unique_nodes_list)}
        num_unique_nodes = len(unique_nodes_list)

        # 2. Process each sequence in the batch
        for k, u_input_original in enumerate(inputs_slice):
             # Map original sequence nodes to new batch-local indices
             u_input_mapped = [node_map.get(item, 0) for item in u_input_original] # Default to 0 if node not in map (shouldn't happen)
             alias_inputs.append(u_input_mapped)

             # `items` should contain the mapped unique nodes for this batch graph
             # We only need one `items` list for the whole batch graph structure
             # We only need one `A` matrix for the whole batch graph structure? No, GNN expects batch dim.
             # The original code builds A per session. Let's stick to that for now.

             # --- Build Graph Adjacency (A) per session ---
             session_nodes = sorted([node_map[item] for item in u_input_original if item in node_map]) # Mapped nodes in this session
             session_node_map = {node_id: idx for idx, node_id in enumerate(session_nodes)} # Map within session mapped nodes
             max_n_node_session = len(session_nodes) # Number of unique nodes in this session

             items.append(session_nodes + [0] * (num_unique_nodes - max_n_node_session)) # Pad `items` to batch max unique nodes

             u_A = np.zeros((max_n_node_session, max_n_node_session), dtype=np.float32)
             for j in range(len(u_input_mapped) - 1):
                  u_mapped = u_input_mapped[j]
                  v_mapped = u_input_mapped[j+1]
                  # Check if they are valid nodes within the session's unique nodes
                  if u_mapped in session_node_map and v_mapped in session_node_map:
                       u_session_idx = session_node_map[u_mapped]
                       v_session_idx = session_node_map[v_mapped]
                       u_A[u_session_idx][v_session_idx] = 1


             # Normalize A (In/Out degrees)
             u_sum_in = np.sum(u_A, 0)
             u_sum_in[np.where(u_sum_in == 0)] = 1 # Avoid division by zero
             u_A_in = np.divide(u_A, u_sum_in)

             u_sum_out = np.sum(u_A, 1)
             u_sum_out[np.where(u_sum_out == 0)] = 1
             u_A_out = np.divide(u_A.transpose(), u_sum_out).transpose() # Correct transpose for out-degree normalization


             # Pad u_A_in and u_A_out to the batch's max_n_node size (num_unique_nodes)
             padded_A_in = np.zeros((num_unique_nodes, num_unique_nodes), dtype=np.float32)
             padded_A_out = np.zeros((num_unique_nodes, num_unique_nodes), dtype=np.float32)
             if max_n_node_session > 0: # Only copy if there are nodes
                 padded_A_in[:max_n_node_session, :max_n_node_session] = u_A_in
                 padded_A_out[:max_n_node_session, :max_n_node_session] = u_A_out


             # Concatenate normalized matrices
             A_combined = np.concatenate([padded_A_in, padded_A_out], axis=1) # Shape: (num_unique, 2 * num_unique)
             A.append(A_combined)


        # Return batch data
        # items: mapped unique nodes for each session graph (padded)
        # A: list of adjacency matrices (one per session)
        # alias_inputs: sequences with nodes mapped to session graph indices
        # mask_slice: original mask for sequences
        # targets_slice: original targets

        # Convert lists to numpy arrays before returning
        # Important: items sent to model embedding should be ORIGINAL node IDs if embedding is global
        # Let's adjust: 'items' should be the unique ORIGINAL nodes in the batch map
        # 'alias_inputs' should map original sequence items to the index in the 'items' list

        # --- Recalculate items and alias_inputs based on ORIGINAL IDs ---
        items_original = unique_nodes_list # List of unique original node IDs in the batch
        alias_inputs_original_map = []
        node_original_to_batch_idx = {node_id: idx for idx, node_id in enumerate(items_original)}

        for u_input_original in inputs_slice:
             # Map original item IDs to their index in the 'items_original' list
             mapped_seq = [node_original_to_batch_idx.get(item, 0) for item in u_input_original] # Default to 0 (padding index)
             alias_inputs_original_map.append(mapped_seq)


        # Return final slice data
        return (
            np.array(alias_inputs_original_map), # Sequences mapped to indices in items_original
            np.array(A),                         # Adjacency matrices for each session graph (using local session indices)
            np.array(items_original),            # Unique ORIGINAL node IDs in the batch
            mask_slice,                          # Original sequence mask
            targets_slice                        # Original targets
        )