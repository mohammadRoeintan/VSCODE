# model.py


import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import copy
from torch.cuda.amp import autocast, GradScaler # For mixed-precision training
import pytz
import utils
import argparse # <<<<<<<<<<< ADD IMPORT ARGPARSE HERE <<<<<<<<<<<

IR_TIMEZONE = pytz.timezone('Asia/Tehran') # For logging time

class GlobalGCN(Module):
    """
    A simple GCN layer for the global graph with a sparse matrix.
    A_hat * X * W
    A_hat: Normalized global sparse adjacency matrix (N, N)
    X: Input item embeddings (N, D_in)
    W: Learnable weight matrix (D_in, D_out)
    Output: (N, D_out)
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GlobalGCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj_sparse_matrix_normalized):
        # x: (N, in_features), adj_sparse_matrix_normalized: (N, N) sparse
        
        # Linear transformation: X * W
        support = self.linear(x)  # Potentially (N, out_features) in Half under autocast

        # --- Ensure inputs to torch.sparse.mm are float32 ---
        current_adj = adj_sparse_matrix_normalized
        current_support = support

        # ** Safeguard: Coalesce adj if it's sparse and not coalesced **
        if current_adj.is_sparse and not current_adj.is_coalesced():
            current_adj = current_adj.coalesce()

        # 1. Ensure the dense matrix ('support') is float32
        if current_support.dtype == torch.half:
            current_support = current_support.float()

        # 2. Ensure the sparse matrix ('adj_sparse_matrix_normalized') operates with float32 values.
        if current_adj.is_sparse:
            if current_adj.values().dtype == torch.half: # This was the error location
                current_adj = torch.sparse_coo_tensor(
                    current_adj.indices(),
                    current_adj.values().float(), # Cast values to float32
                    current_adj.size(),
                    dtype=torch.float32, # Explicitly set dtype for the new sparse tensor
                    device=current_adj.device
                ).coalesce() # Coalesce after potential recreation
        elif current_adj.dtype == torch.half: # Fallback if somehow it became dense and half
             current_adj = current_adj.float()
        # --- End of float32 enforcement ---
        
        # Perform sparse matrix multiplication with float32 tensors
        output = torch.sparse.mm(current_adj, current_support) # A_hat * X * W
        
        return output


class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe shape is (max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # self.pe is (max_len, d_model). We need (seq_len, d_model) part.
        # Broadcasting will handle adding (seq_len, d_model) to (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :] 
        return self.dropout(x)


class GNN(Module): # For Local Session Graph
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size)) # Bias for input_in after matmul
        self.b_oah = Parameter(torch.Tensor(self.hidden_size)) # Bias for input_out after matmul
        
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        self.reset_parameters() # Parameter initialization

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        # Initialize larger matrices with Xavier
        if self.w_ih.dim() > 1: nn.init.xavier_uniform_(self.w_ih)
        if self.w_hh.dim() > 1: nn.init.xavier_uniform_(self.w_hh)
        # Initialize biases and smaller params with uniform
        nn.init.uniform_(self.b_ih, -stdv, stdv)
        nn.init.uniform_(self.b_hh, -stdv, stdv)
        nn.init.uniform_(self.b_iah, -stdv, stdv)
        nn.init.uniform_(self.b_oah, -stdv, stdv)
        # Initialize linear layers within GNN (their reset_parameters will be called by nn.Linear)

    def GNNCell(self, A, hidden): # A is (B, max_nodes, 2*max_nodes), hidden is (B, max_nodes, D)
        # A_in: (B, max_nodes, max_nodes), A_out: (B, max_nodes, max_nodes)
        A_in = A[:, :, :A.size(1)] # Adjacency matrix for incoming edges
        A_out = A[:, :, A.size(1): 2 * A.size(1)] # Adjacency matrix for outgoing edges
        
        # Messages from incoming edges
        input_in = torch.matmul(A_in, self.linear_edge_in(hidden)) + self.b_iah 
        # Messages from outgoing edges
        input_out = torch.matmul(A_out, self.linear_edge_out(hidden)) + self.b_oah
        
        # Concatenate messages
        inputs = torch.cat([input_in, input_out], 2) # (B, max_nodes, 2*D)
        
        # Gated Recurrent Unit (GRU-like) gates
        gi = F.linear(inputs, self.w_ih, self.b_ih) # (B, max_nodes, 3*D)
        gh = F.linear(hidden, self.w_hh, self.b_hh) # (B, max_nodes, 3*D)
        
        i_r, i_i, i_n = gi.chunk(3, 2) # reset, input, new gates from input
        h_r, h_i, h_n = gh.chunk(3, 2) # reset, input, new gates from hidden
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n) # New candidate hidden state
        
        hy = newgate + inputgate * (hidden - newgate) # Final hidden state
        return hy

    def forward(self, A, hidden):
        for _ in range(self.step): # Propagate for 'step' iterations
            hidden = self.GNNCell(A, hidden)
        return hidden


class TargetAwareEncoderLayer(Module): # Standard Transformer Encoder Layer
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super(TargetAwareEncoderLayer, self).__init__()
        # Using batch_first=True for MultiheadAttention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src, src_key_padding_mask=None): # src: (Batch, Seq, Feature)
        src_norm = self.norm1(src)
        # MultiheadAttention expects query, key, value. For self-attention, they are the same.
        # attn_mask is for preventing attention to future tokens (decoder) or specific positions.
        # src_key_padding_mask (Batch, Seq) True for padded tokens.
        sa_output, _ = self.self_attn(src_norm, src_norm, src_norm,
                                    attn_mask=None, 
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(sa_output) # Add & Norm (residual connection first)
        
        src_norm = self.norm2(src) # Norm before FFN
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_output) # Add & Norm
        return src


class TargetAwareTransformerEncoder(Module): # Standard Transformer Encoder
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TargetAwareTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm # Final LayerNorm after all layers

    def forward(self, src, mask=None, src_key_padding_mask=None): # mask is typically for attention, not used here
        output = src
        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class SessionGraph(Module):
    def __init__(self, opt, n_node, global_adj_sparse_matrix=None):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.nonhybrid = opt.nonhybrid
        self.ssl_weight = opt.ssl_weight
        self.ssl_temp = opt.ssl_temp
        self.ssl_dropout_rate = opt.ssl_dropout_rate
        self.num_global_gcn_layers_config = opt.global_gcn_layers
        
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)
        
        self.use_global_graph = False
        self.global_gcn_layers_module = None 
        if self.num_global_gcn_layers_config > 0 and global_adj_sparse_matrix is not None:
            if not global_adj_sparse_matrix.is_sparse:
                print("Warning: Global adjacency matrix received by SessionGraph is not sparse. Coalescing.")
                global_adj_sparse_matrix = global_adj_sparse_matrix.to_sparse_coo().coalesce()
            elif not global_adj_sparse_matrix.is_coalesced():
                print("Warning: Global adjacency matrix received by SessionGraph is sparse but not coalesced. Coalescing.")
                global_adj_sparse_matrix = global_adj_sparse_matrix.coalesce()

            self.register_buffer('global_adj_sparse_matrix_normalized', global_adj_sparse_matrix)
            
            self.global_gcn_layers_module = nn.ModuleList()
            for _ in range(self.num_global_gcn_layers_config):
                # Assuming hidden_size remains consistent through GCN layers
                self.global_gcn_layers_module.append(GlobalGCN(self.hidden_size, self.hidden_size))
            self.use_global_graph = True
            print(f"SessionGraph: Using {self.num_global_gcn_layers_config} global GCN layers. Sparse adj matrix shape: {self.global_adj_sparse_matrix_normalized.shape}, nnz: {self.global_adj_sparse_matrix_normalized._nnz()}")
        else:
            self.global_adj_sparse_matrix_normalized = None
            print("SessionGraph: Global graph processing is disabled.")

        self.gnn_local = GNN(self.hidden_size, step=opt.step)
        self.pos_encoder = PositionalEncoding(self.hidden_size, getattr(opt, 'dropout', 0.1))
        
        ta_encoder_layer = TargetAwareEncoderLayer(
            d_model=self.hidden_size,
            nhead=getattr(opt, 'nhead', 2),
            dim_feedforward=getattr(opt, 'ff_hidden', 256),
            dropout=getattr(opt, 'dropout', 0.1)
        )
        self.transformer_encoder = TargetAwareTransformerEncoder(
            encoder_layer=ta_encoder_layer,
            num_layers=getattr(opt, 'nlayers', 2),
            norm=nn.LayerNorm(self.hidden_size)
        )
        
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False) # For attention scores
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True) # For non-hybrid
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # For hybrid target attention
        
        self.loss_function = nn.CrossEntropyLoss()
        # Optimizer and Scheduler are initialized in main.py and assigned as model attributes
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if not param.requires_grad: continue # Skip non-trainable params like buffers

            # More specific initialization based on common practices
            if 'weight_ih' in name or 'weight_hh' in name or \
               (hasattr(param, 'is_leaf') and param.is_leaf and param.dim() > 1 and 'embedding' not in name): # Linear/GNN weights
                nn.init.xavier_uniform_(param)
            elif 'bias' in name or (param.dim() == 1 and 'weight' not in name): # Biases and 1D params
                nn.init.uniform_(param, -stdv, stdv)
            elif 'embedding.weight' in name: # Embedding weights (except padding)
                nn.init.normal_(param, mean=0, std=0.01) # A common init for embeddings

        with torch.no_grad(): # Ensure padding embedding is zero
            if self.embedding.padding_idx is not None:
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        
        # Initialize GlobalGCN layers' linear weights (biases are uniform by default if enabled)
        if self.global_gcn_layers_module is not None:
            for gcn_layer in self.global_gcn_layers_module:
                if hasattr(gcn_layer, 'linear') and hasattr(gcn_layer.linear, 'weight'):
                     nn.init.xavier_uniform_(gcn_layer.linear.weight)


    def _get_enriched_item_embeddings(self):
        all_item_initial_embeddings = self.embedding.weight # (n_node, D)
        
        if self.use_global_graph and self.global_gcn_layers_module is not None:
            current_embeddings = all_item_initial_embeddings
            # global_adj_sparse_matrix_normalized is a buffer, should be on the correct device
            adj = self.global_adj_sparse_matrix_normalized 
            
            for i, gcn_layer in enumerate(self.global_gcn_layers_module):
                current_embeddings = gcn_layer(current_embeddings, adj)
                current_embeddings = F.relu(current_embeddings) 
                # Optional: Add dropout between GCN layers
                # current_embeddings = F.dropout(current_embeddings, p=0.5, training=self.training)
            
            # Option: Residual connection to initial embeddings
            # return all_item_initial_embeddings + current_embeddings 
            return current_embeddings # Using output of last GCN layer
        else:
            return all_item_initial_embeddings


    def _process_session_graph_local(self, items_local_session_ids, A_local_session_adj, enriched_all_item_embeddings):
        # Get enriched embeddings for items specific to current sessions
        hidden_local_session_enriched = F.embedding(
            items_local_session_ids, # (B, max_nodes_in_batch)
            enriched_all_item_embeddings, # (n_node, D)
            padding_idx=0 # If item_id 0 is used for padding in items_local_session_ids
        ) # Output: (B, max_nodes_in_batch, D)
        
        # Process with local GNN
        hidden_local_session_processed = self.gnn_local(A_local_session_adj, hidden_local_session_enriched)
        return hidden_local_session_processed # (B, max_nodes_in_batch, D)


    def compute_scores(self, hidden_transformer_output, mask_for_seq, all_item_embeddings_for_scoring):
        # hidden_transformer_output: (B, L, D)
        # mask_for_seq: (B, L)
        # all_item_embeddings_for_scoring: (n_node, D)
        
        mask_float = mask_for_seq.float() # Ensure mask is float for operations like sum
        
        # Get hidden state of the last valid item in each sequence
        batch_indices = torch.arange(mask_float.size(0), device=hidden_transformer_output.device)
        last_item_indices = torch.clamp(mask_float.sum(1) - 1, min=0).long().to(hidden_transformer_output.device)
        ht = hidden_transformer_output[batch_indices, last_item_indices] # (B, D), representation of last item

        # Attention mechanism to get session preference 'a'
        q1 = self.linear_one(ht).unsqueeze(1) # (B, 1, D)
        q2 = self.linear_two(hidden_transformer_output) # (B, L, D)
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1) # (B, L)
        
        # Apply mask to attention logits before softmax
        alpha_logits_masked = alpha_logits.masked_fill(mask_for_seq == 0, torch.finfo(alpha_logits.dtype).min)
        alpha = torch.softmax(alpha_logits_masked, dim=1) # (B, L), attention weights
        
        # Weighted sum of item representations in the sequence
        # mask_float.unsqueeze(-1) ensures padded items don't contribute: (B, L, 1)
        a = (alpha.unsqueeze(-1) * hidden_transformer_output * mask_float.unsqueeze(-1)).sum(1) # (B, D), session preference
        
        # Candidate item embeddings (excluding padding item 0)
        candidate_embeds = all_item_embeddings_for_scoring[1:]  # (n_node-1, D)

        if self.nonhybrid:
            # Combine session preference 'a' and last item 'ht'
            combined_session_rep = self.linear_transform(torch.cat([a, ht], dim=1)) # (B, D)
            scores = torch.matmul(combined_session_rep, candidate_embeds.t()) # (B, n_node-1)
        else: # Hybrid mode with target attention (TAGNN style)
            qt = self.linear_t(hidden_transformer_output) # (B, L, D), transformed sequence for target attention
            
            # Calculate similarity between candidates and sequence items (for target attention)
            # candidate_embeds: (num_cand, D), qt.transpose: (B, D, L) -> beta_logits: (B, num_cand, L)
            beta_logits = torch.matmul(candidate_embeds, qt.transpose(1, 2))
            
            # Apply mask to beta_logits (mask_for_seq.unsqueeze(1) -> (B, 1, L))
            beta_logits_masked = beta_logits.masked_fill(mask_for_seq.unsqueeze(1) == 0, torch.finfo(beta_logits.dtype).min)
            beta = torch.softmax(beta_logits_masked, dim=-1) # (B, num_cand, L), attention weights over sequence items for each candidate
            
            # Context vector for each candidate based on attended sequence items
            # (qt * mask_float.unsqueeze(-1)) zeros out padded items in qt
            target_ctx = torch.matmul(beta, qt * mask_float.unsqueeze(-1)) # (B, num_cand, D)
            
            # Combine session preference 'a' (broadcasted) with target context
            final_representation = a.unsqueeze(1) + target_ctx # a.unsqueeze(1): (B, 1, D)
            
            # Final scores by dot product with candidate embeddings
            # candidate_embeds.unsqueeze(0): (1, num_cand, D)
            scores = torch.sum(final_representation * candidate_embeds.unsqueeze(0), dim=-1) # (B, num_cand)
        
        return scores


    def forward_model_logic(self, alias_inputs_local_ids, A_local_adj, items_local_ids, mask_for_seq, is_train=True):
        # 1. Get (globally) enriched item embeddings
        enriched_all_item_embeddings = self._get_enriched_item_embeddings() # (n_node, D)

        # 2. Process local session graph using enriched embeddings
        hidden_session_items_processed = self._process_session_graph_local(
            items_local_ids, A_local_adj, enriched_all_item_embeddings
        ) # (B, max_nodes_in_batch, D)
        
        # 3. Gather item representations according to original sequence order
        # alias_inputs_local_ids: (B, L), maps positions in sequence to local GNN output indices
        # unsqueeze to (B, L, 1) and expand to (B, L, D) for gather
        seq_hidden_gnn_output = torch.gather(
            hidden_session_items_processed, 
            dim=1, 
            index=alias_inputs_local_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        ) # (B, L, D)
        
        # 4. Apply Positional Encoding
        seq_hidden_with_pos = self.pos_encoder(seq_hidden_gnn_output) # (B, L, D)
        
        # 5. Pass through Transformer Encoder
        # src_key_padding_mask is True for padded tokens
        src_key_padding_mask = (mask_for_seq == 0) 
        output_transformer = self.transformer_encoder(
            src=seq_hidden_with_pos,
            src_key_padding_mask=src_key_padding_mask
        ) # (B, L, D)
        
        # 6. Compute final scores for candidate items
        scores = self.compute_scores(output_transformer, mask_for_seq, enriched_all_item_embeddings)

        # 7. SSL Loss Calculation (optional)
        ssl_loss_value = torch.tensor(0.0, device=scores.device) # Default to 0
        if is_train and self.ssl_weight > 0:
            try:
                # Use GNN output before Transformer for SSL, as it's closer to item structure
                last_idx_for_ssl = torch.clamp(mask_for_seq.sum(1) - 1, min=0).long()
                batch_indices_ssl = torch.arange(mask_for_seq.size(0), device=seq_hidden_gnn_output.device)
                last_idx_for_ssl = last_idx_for_ssl.to(seq_hidden_gnn_output.device) # Ensure device match

                ssl_base_emb_seq = seq_hidden_gnn_output[batch_indices_ssl, last_idx_for_ssl]
                
                ssl_emb1 = F.dropout(ssl_base_emb_seq, p=self.ssl_dropout_rate, training=True)
                ssl_emb2 = F.dropout(ssl_base_emb_seq, p=self.ssl_dropout_rate, training=True)
                ssl_loss_value = self.calculate_ssl_loss(ssl_emb1, ssl_emb2, self.ssl_temp)
            except Exception as e:
                print(f"SSL calculation error: {e}") # Log error, loss remains 0
        
        return scores, ssl_loss_value

    def calculate_ssl_loss(self, emb1, emb2, temperature): # Standard InfoNCE loss
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        # Similarity matrix between two sets of embeddings
        sim_matrix_12 = torch.matmul(emb1_norm, emb2_norm.t()) / temperature
        # Log-softmax over columns (predicting emb2 from emb1)
        log_softmax_12 = F.log_softmax(sim_matrix_12, dim=1)
        # Loss for emb1 -> emb2 (diagonal elements are positive pairs)
        loss_12 = -torch.diag(log_softmax_12) 
        
        # Symmetric loss: emb2 -> emb1
        sim_matrix_21 = torch.matmul(emb2_norm, emb1_norm.t()) / temperature
        log_softmax_21 = F.log_softmax(sim_matrix_21, dim=1)
        loss_21 = -torch.diag(log_softmax_21)
        
        return (loss_12.mean() + loss_21.mean()) / 2.0


# --- Wrapper for train_test compatibility ---
def forward(model: SessionGraph, i_batch_indices, data_loader: utils.Data, is_train=True):
    # Get batch data using indices
    alias_inputs_np, A_local_np, items_local_np, mask_seq_np, targets_np = data_loader.get_slice(i_batch_indices)
    
    # Convert numpy arrays to tensors and move to model's device
    current_device = next(model.parameters()).device
    
    alias_inputs = torch.from_numpy(alias_inputs_np).long().to(current_device)
    A_local_adj = torch.from_numpy(A_local_np).float().to(current_device)
    items_local_ids = torch.from_numpy(items_local_np).long().to(current_device)
    mask_for_seq = torch.from_numpy(mask_seq_np).float().to(current_device) # Mask is float for some ops
    targets = torch.from_numpy(targets_np).long().to(current_device)
    
    # Call the main model logic
    scores, ssl_loss = model.forward_model_logic(
        alias_inputs, 
        A_local_adj, 
        items_local_ids, 
        mask_for_seq, 
        is_train=is_train
    )
    
    return targets, scores, ssl_loss


def train_test(model: SessionGraph, train_data: utils.Data, eval_data: utils.Data, opt: argparse.Namespace):
    use_amp = torch.cuda.is_available() # Enable AMP if CUDA is available
    scaler = GradScaler(enabled=use_amp) 
    
    # --- Training Phase ---
    model.train() # Set model to training mode
    total_loss_epoch = 0.0
    total_rec_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0
    
    train_batch_slices = train_data.generate_batch(opt.batchSize)
    num_train_batches = len(train_batch_slices)

    if num_train_batches == 0:
        print("Warning: No batches to train on in train_data.")
    else:
        # Log start of training for the epoch
        # current_epoch_display = model.scheduler.last_epoch +1 if hasattr(model, 'scheduler') else 'N/A'
        # print(f'Starting Training for Epoch {current_epoch_display}...')

        for step, batch_indices in enumerate(train_batch_slices):
            if not hasattr(model, 'optimizer'): # Should be set in main.py
                print("CRITICAL ERROR: model.optimizer is not set!")
                return 0.0, 0.0, 0.0 
            
            model.optimizer.zero_grad(set_to_none=True) # More efficient zeroing
            
            with autocast(enabled=use_amp): # AMP context
                targets, scores, ssl_loss = forward(model, batch_indices, train_data, is_train=True)
                
                # Calculate recommendation loss
                valid_targets_mask = (targets > 0) & (targets < model.n_node) # Targets are 1-based
                rec_loss = torch.tensor(0.0, device=scores.device)
                if valid_targets_mask.any():
                    # Scores are for items 1 to n_node-1 (0-indexed if padding is 0)
                    # Targets (1 to n_node) need to be mapped to 0 to n_node-1 for loss
                    target_values_0_based = (targets[valid_targets_mask] - 1).clamp(0, scores.size(1) - 1)
                    rec_loss = model.loss_function(scores[valid_targets_mask], target_values_0_based)
                
                current_batch_loss = rec_loss + model.ssl_weight * ssl_loss
            
            # Backward pass and optimizer step
            if use_amp:
                scaler.scale(current_batch_loss).backward()
                scaler.step(model.optimizer)
                scaler.update()
            else:
                current_batch_loss.backward()
                model.optimizer.step()
            
            total_loss_epoch += current_batch_loss.item()
            total_rec_loss_epoch += rec_loss.item()
            total_ssl_loss_epoch += ssl_loss.item() if isinstance(ssl_loss, torch.Tensor) else ssl_loss
            
            if (step + 1) % max(1, num_train_batches // 5) == 0 or step == num_train_batches - 1:
                avg_total_loss = total_loss_epoch / (step + 1)
                avg_rec_loss = total_rec_loss_epoch / (step + 1)
                avg_ssl_loss = total_ssl_loss_epoch / (step + 1)
                # current_epoch_display = model.scheduler.last_epoch +1 if hasattr(model, 'scheduler') else 'N/A' # For display
                print(f'  Training Batch [{step + 1}/{num_train_batches}] '
                      f'Avg Total Loss: {avg_total_loss:.4f}, Avg Rec Loss: {avg_rec_loss:.4f}, Avg SSL Loss: {avg_ssl_loss:.4f}')
    
    if hasattr(model, 'scheduler'): # Step the LR scheduler after each epoch
        model.scheduler.step()
    
    # --- Evaluation Phase ---
    model.eval() # Set model to evaluation mode
    k_metric = opt.k_metric 
    
    # Default return values if no evaluation data
    final_recall_at_k, final_mrr_at_k, final_precision_at_k = 0.0, 0.0, 0.0

    if eval_data is None or eval_data.length == 0:
        print("No evaluation data provided or evaluation data is empty. Skipping evaluation metrics calculation.")
        return final_recall_at_k, final_mrr_at_k, final_precision_at_k

    hit_at_k, mrr_at_k = [], [] # precision_at_k can be added if needed
    eval_batch_slices = eval_data.generate_batch(opt.batchSize)

    if not eval_batch_slices:
        print("No batches to evaluate on in eval_data.")
        return final_recall_at_k, final_mrr_at_k, final_precision_at_k

    with torch.no_grad(): # Disable gradient calculations for evaluation
        for batch_indices_eval in eval_batch_slices:
            targets_eval, scores_eval, _ = forward(model, batch_indices_eval, eval_data, is_train=False)
            
            # Get top-k predictions
            # scores_eval shape: (B, num_candidates) where num_candidates = n_node-1
            _, top_k_indices_0_based = scores_eval.topk(k_metric, dim=1) # (B, k)
            
            # Convert 0-based indices (0 to n_node-2) back to 1-based item IDs (1 to n_node-1)
            top_k_item_ids = top_k_indices_0_based + 1 
            
            targets_eval_np = targets_eval.cpu().numpy() # (B,)
            top_k_item_ids_np = top_k_item_ids.cpu().numpy() # (B, k)

            for i in range(targets_eval_np.shape[0]):
                target_item_id = targets_eval_np[i]
                predicted_k_item_ids = top_k_item_ids_np[i]
                
                # Consider only valid targets (item_id > 0 and < n_node)
                if target_item_id > 0 and target_item_id < model.n_node:
                    if target_item_id in predicted_k_item_ids:
                        hit_at_k.append(1)
                        # Find rank of the target item in predictions
                        rank = np.where(predicted_k_item_ids == target_item_id)[0][0] + 1
                        mrr_at_k.append(1.0 / rank)
                    else:
                        hit_at_k.append(0)
                        mrr_at_k.append(0.0)
                    # Precision@k can be calculated here if needed
    
    if hit_at_k: final_recall_at_k = np.mean(hit_at_k) * 100
    if mrr_at_k: final_mrr_at_k = np.mean(mrr_at_k) * 100
    
    print(f'Evaluation Results @{k_metric}: Recall: {final_recall_at_k:.4f}%, MRR: {final_mrr_at_k:.4f}%')
    
    return final_recall_at_k, final_mrr_at_k, final_precision_at_k # Precision is 0 for now