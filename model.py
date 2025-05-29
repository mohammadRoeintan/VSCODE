# model.py

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import copy
from torch.cuda.amp import autocast, GradScaler # For mixed precision training
import pytz
import utils
import argparse

IR_TIMEZONE = pytz.timezone('Asia/Tehran')

class GlobalGCN(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GlobalGCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj_sparse_matrix_normalized):
        # x should be float32, adj_sparse_matrix_normalized should be float32 and coalesced
        support = self.linear(x) 
        output = torch.sparse.mm(adj_sparse_matrix_normalized, support)
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# --- SAGEConv Layer (GraphSAGE Convolution) ---
class SAGEConv(Module):
    def __init__(self, in_features, out_features, aggregator_type='mean', bias=True):
        super(SAGEConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator_type = aggregator_type.lower()

        # Weight matrix for current node's own features (W_self * h_v)
        self.fc_self = nn.Linear(in_features, out_features, bias=bias)
        # Weight matrix for aggregated_neighbors features (W_neigh * h_N(v))
        self.fc_neigh = nn.Linear(in_features, out_features, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_self.weight)
        if self.fc_self.bias is not None:
            nn.init.zeros_(self.fc_self.bias)
        nn.init.xavier_uniform_(self.fc_neigh.weight)
        if self.fc_neigh.bias is not None:
            nn.init.zeros_(self.fc_neigh.bias)

    def forward(self, x, adj):
        # x: node features (batch_size, num_nodes, in_features)
        # adj: adjacency matrix (batch_size, num_nodes, num_nodes), 0/1 indicating connectivity
        
        # Aggregate neighbor features
        # adj is (B, N, N), x is (B, N, F)
        # torch.bmm(adj, x) computes sum(A_ij * x_j) for each i. If A_ij is 0/1, this is sum of neighbor features.
        sum_neighbor_features = torch.bmm(adj, x) # (B, N, F)
        
        if self.aggregator_type == 'mean':
            num_neighbors = adj.sum(dim=2, keepdim=True).clamp(min=1.0) # (B, N, 1), count of neighbors
            aggregated_neighbors = sum_neighbor_features / num_neighbors
        # elif self.aggregator_type == 'pool': # Max pooling would require more complex logic for arbitrary neighborhoods
        #     # For a proper max-pool, one might need to gather neighbors first and then apply max.
        #     # This is a simplification if only positive features and 0/1 adj:
        #     # masked_x_for_pool = x.unsqueeze(1).expand(-1, x.size(1), -1, -1) * adj.unsqueeze(-1) # B,N,N,F
        #     # aggregated_neighbors, _ = masked_x_for_pool.max(dim=2) # B,N,F
        #     # This simplistic pool doesn't handle no-neighbor cases well without more masking.
        #     print("Warning: 'pool' aggregator is complex; 'mean' is typically used or a library like PyG.")
        #     num_neighbors = adj.sum(dim=2, keepdim=True).clamp(min=1.0) 
        #     aggregated_neighbors = sum_neighbor_features / num_neighbors # Fallback to mean
        else: # Default to mean aggregator
            if self.aggregator_type != 'mean':
                 print(f"Warning: Aggregator type '{self.aggregator_type}' not fully implemented or recognized. Using 'mean'.")
            num_neighbors = adj.sum(dim=2, keepdim=True).clamp(min=1.0)
            aggregated_neighbors = sum_neighbor_features / num_neighbors

        # Transform self features
        self_feat_transformed = self.fc_self(x) # (B, N, out_features)
        
        # Transform aggregated neighbor features
        neigh_feat_transformed = self.fc_neigh(aggregated_neighbors) # (B, N, out_features)
        
        # Combine transformed features (element-wise sum is common, followed by activation)
        # h_v_new = W_self * h_v + W_neigh * aggregate(h_u for u in N(v))
        output = self_feat_transformed + neigh_feat_transformed
        
        return F.relu(output) # Apply non-linearity

# --- GraphSAGE GNN (stacks multiple SAGEConv layers) ---
class GraphSAGE_GNN(Module):
    def __init__(self, hidden_size, num_layers=1, aggregator_type='mean'):
        super(GraphSAGE_GNN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type=aggregator_type))
        self.num_layers = num_layers
        # Optional: Add LayerNorm after each SAGEConv or at the end of all layers
        # self.norm = nn.LayerNorm(hidden_size) 

    def forward(self, x, adj):
        # x: initial node features (batch_size, num_nodes, hidden_size)
        # adj: adjacency matrix (batch_size, num_nodes, num_nodes)
        h = x
        for layer in self.layers:
            h = layer(h, adj)
            # h = self.norm(h) # Example of applying LayerNorm
        return h

class TargetAwareEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super(TargetAwareEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src, src_key_padding_mask=None):
        # src shape: (batch_size, seq_len, d_model)
        src_norm = self.norm1(src)
        sa_output, _ = self.self_attn(src_norm, src_norm, src_norm,
                                    attn_mask=None, # Self-attention, no causal mask here unless specified
                                    key_padding_mask=src_key_padding_mask) # (batch_size, seq_len)
        src = src + self.dropout1(sa_output)
        src_norm = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_output)
        return src

class TargetAwareTransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TargetAwareTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm # Final LayerNorm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        return output

class SessionGraph(Module):
    def __init__(self, opt: argparse.Namespace, n_node, global_adj_sparse_matrix=None):
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
            current_adj_for_buffer = global_adj_sparse_matrix
            # Ensure it's sparse COO and coalesced (critical for sparse_mm)
            if not current_adj_for_buffer.is_sparse:
                current_adj_for_buffer = current_adj_for_buffer.to_sparse_coo()
            if not current_adj_for_buffer.is_coalesced():
                current_adj_for_buffer = current_adj_for_buffer.coalesce()
            
            # Ensure dtype is float32 for GCN operations if it's not already
            if current_adj_for_buffer.values().dtype != torch.float32:
                current_adj_for_buffer = torch.sparse_coo_tensor(
                    current_adj_for_buffer.indices(),
                    current_adj_for_buffer.values().float(),
                    current_adj_for_buffer.size(),
                    dtype=torch.float32,
                    device=current_adj_for_buffer.device # Keep original device
                ).coalesce()

            self.register_buffer('global_adj_sparse_matrix_normalized', current_adj_for_buffer)

            self.global_gcn_layers_module = nn.ModuleList()
            for _ in range(self.num_global_gcn_layers_config):
                self.global_gcn_layers_module.append(GlobalGCN(self.hidden_size, self.hidden_size))
            self.use_global_graph = True
            print(f"SessionGraph: Using {self.num_global_gcn_layers_config} global GCN layers. "
                  f"Sparse adj matrix shape: {self.global_adj_sparse_matrix_normalized.shape}, "
                  f"nnz: {self.global_adj_sparse_matrix_normalized._nnz()}, "
                  f"values_dtype: {self.global_adj_sparse_matrix_normalized.values().dtype}")
        else:
            self.global_adj_sparse_matrix_normalized = None # Explicitly set to None
            print("SessionGraph: Global graph processing is disabled.")

        # --- Replace GNN with GraphSAGE_GNN ---
        aggregator = getattr(opt, 'aggregator_type', 'mean') # Allow specifying aggregator via opt
        self.gnn_local = GraphSAGE_GNN(self.hidden_size, num_layers=opt.step, aggregator_type=aggregator)
        print(f"SessionGraph: Using GraphSAGE for local GNN with {opt.step} layers and '{aggregator}' aggregator.")
        
        self.pos_encoder = PositionalEncoding(self.hidden_size, getattr(opt, 'dropout', 0.1))

        # Transformer Encoder
        ta_encoder_layer = TargetAwareEncoderLayer(
            d_model=self.hidden_size,
            nhead=getattr(opt, 'nhead', 2), # Default nhead if not in opt
            dim_feedforward=getattr(opt, 'ff_hidden', 256), # Default ff_hidden
            dropout=getattr(opt, 'dropout', 0.1) # Default dropout
        )
        self.transformer_encoder = TargetAwareTransformerEncoder(
            encoder_layer=ta_encoder_layer,
            num_layers=getattr(opt, 'nlayers', 2), # Default nlayers
            norm=nn.LayerNorm(self.hidden_size) # Final norm for transformer output
        )

        # Linear layers for scoring and attention
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # For target-aware attention

        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if not param.requires_grad: 
                continue
            
            if 'embedding.weight' == name:
                 nn.init.normal_(param, mean=0, std=0.01)
                 if self.embedding.padding_idx is not None:
                     with torch.no_grad():
                         param[self.embedding.padding_idx].fill_(0)
            # General Xavier for weights, excluding specific modules handled elsewhere
            elif (param.dim() > 1 and 'weight' in name and
                  'global_gcn_layers_module' not in name and # GlobalGCN has its own init pattern
                  'gnn_local' not in name and                # GraphSAGE_GNN has its own init pattern
                  'embedding' not in name):                  # Embedding handled above
                nn.init.xavier_uniform_(param)
            # General uniform for biases or 1D params, excluding specific modules
            elif ('bias' in name or param.dim() == 1) and \
                 ('gnn_local' not in name and 'global_gcn_layers_module' not in name):
                nn.init.uniform_(param, -stdv, stdv)
        
        # Initialize GlobalGCN layers' linear components
        if self.global_gcn_layers_module is not None:
            for gcn_layer in self.global_gcn_layers_module:
                 if hasattr(gcn_layer, 'linear') and hasattr(gcn_layer.linear, 'weight'):
                      nn.init.xavier_uniform_(gcn_layer.linear.weight)
                      if gcn_layer.linear.bias is not None:
                           nn.init.zeros_(gcn_layer.linear.bias)
        
        # Note: GraphSAGE_GNN (self.gnn_local) and its SAGEConv layers handle their own parameter initialization
        # within their respective reset_parameters methods, called during their __init__.

    def _get_enriched_item_embeddings(self):
        all_item_initial_embeddings = self.embedding.weight # Shape: (n_node, hidden_size)

        if self.use_global_graph and self.global_gcn_layers_module is not None and self.global_adj_sparse_matrix_normalized is not None:
            # Global GCN operates in float32 for stability with sparse operations
            with autocast(enabled=False): 
                current_embeddings_float32 = all_item_initial_embeddings.float()
                adj_matrix_for_gcn = self.global_adj_sparse_matrix_normalized 
                
                for gcn_layer in self.global_gcn_layers_module:
                    current_embeddings_float32 = gcn_layer(current_embeddings_float32, adj_matrix_for_gcn)
                    current_embeddings_float32 = F.relu(current_embeddings_float32) 
            # Return float32 embeddings after GCN processing.
            # Subsequent layers (like local GNN or Transformer) will handle dtypes based on AMP context.
            return current_embeddings_float32 
        else:
            # If no global GCN, return embeddings directly. Dtype is typically float32 from nn.Embedding.
            return all_item_initial_embeddings 

    def _process_session_graph_local(self, items_local_session_ids, A_local_session_adj, enriched_all_item_embeddings):
        # Get embeddings for items present in the local session graphs of the current batch
        # items_local_session_ids: (batch_size, max_unique_nodes_in_batch) - unique item IDs in batch sessions
        # enriched_all_item_embeddings: (n_node, hidden_size) - all item embeddings (from global GCN or initial)
        hidden_local_session_enriched = F.embedding(
            items_local_session_ids, 
            enriched_all_item_embeddings,
            padding_idx=0 if self.embedding.padding_idx == 0 else -1 # Match embedding padding_idx if used for items_local_session_ids
        )
        # hidden_local_session_enriched shape: (batch_size, max_unique_nodes_in_batch, hidden_size)
        # A_local_session_adj shape: (batch_size, max_unique_nodes_in_batch, max_unique_nodes_in_batch)
        
        # Pass to GraphSAGE: (features, adjacency_matrix)
        # The dtypes should be consistent (e.g., both float32 if not using AMP, or matching AMP's context)
        hidden_local_session_processed = self.gnn_local(hidden_local_session_enriched, A_local_session_adj)
        return hidden_local_session_processed


    def compute_scores(self, hidden_transformer_output, mask_for_seq, all_item_embeddings_for_scoring):
        # hidden_transformer_output: (B, L, H) - Output of transformer for each item in sequence
        # mask_for_seq: (B, L) - Mask for valid items in sequence
        # all_item_embeddings_for_scoring: (N, H) - Embeddings of all candidate items
        
        mask_float = mask_for_seq.float().unsqueeze(-1) # (B, L, 1) for broadcasting

        # Get hidden state of the last valid item in each sequence
        batch_indices = torch.arange(mask_for_seq.size(0), device=hidden_transformer_output.device)
        # Sum of mask gives sequence length. -1 for 0-based index. Clamp for empty sequences.
        last_item_indices = torch.clamp(mask_for_seq.sum(1) - 1, min=0).long()
        ht = hidden_transformer_output[batch_indices, last_item_indices] # (B, H) Session embedding (last item)
        
        # Attention mechanism for session representation 'a'
        q1 = self.linear_one(ht).unsqueeze(1) # (B, 1, H)
        q2 = self.linear_two(hidden_transformer_output) # (B, L, H)
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1) # (B, L)
        alpha_logits_masked = alpha_logits.masked_fill(mask_for_seq == 0, torch.finfo(alpha_logits.dtype).min) # Apply mask
        alpha = torch.softmax(alpha_logits_masked, dim=1) # (B, L) Attention weights
        
        # Weighted sum of item hidden states based on alpha
        a = (alpha.unsqueeze(-1) * hidden_transformer_output * mask_float).sum(1) # (B, H) Final session representation 'a'
        
        candidate_embeds = all_item_embeddings_for_scoring[1:] # Exclude padding item 0. Shape: (N_items-1, H)

        if self.nonhybrid: # Only use global preference 'a' and last item 'ht'
            combined_session_rep = self.linear_transform(torch.cat([a, ht], dim=1)) # (B, H_combined) -> (B, H)
            scores = torch.matmul(combined_session_rep, candidate_embeds.t()) # (B, N_items-1)
        else: # Hybrid: combines global session preference 'a' with target-aware attention
            qt = self.linear_t(hidden_transformer_output) # (B, L, H), transformed sequence for target attention
            
            # Calculate scores for each candidate item against each item in the sequence context
            # candidate_embeds: (N_c, H), qt: (B, L, H)
            # We want beta_logits: (B, N_c, L)
            beta_logits = torch.einsum('ch,blh->bcl', candidate_embeds, qt) 

            beta_logits_masked = beta_logits.masked_fill(mask_for_seq.unsqueeze(1) == 0, torch.finfo(beta_logits.dtype).min) # Mask invalid seq positions
            beta = torch.softmax(beta_logits_masked, dim=-1) # (B, N_c, L) Target-specific attention weights
            
            # Compute target-specific context vector for each candidate
            # beta: (B, N_c, L), (qt * mask_float): (B, L, H)
            target_ctx = torch.einsum('bcl,blh->bch', beta, qt * mask_float) # (B, N_c, H)
            
            # Final representation for each candidate: global session 'a' + target-specific context
            final_representation = a.unsqueeze(1) + target_ctx # (B, 1, H) + (B, N_c, H) -> (B, N_c, H)
            
            # Score = sum over hidden dim of (final_representation * candidate_embeds)
            # final_representation: (B, N_c, H), candidate_embeds.unsqueeze(0): (1, N_c, H)
            scores = torch.sum(final_representation * candidate_embeds.unsqueeze(0), dim=-1) # (B, N_c) which is (B, N_items-1)
            
        return scores

    def forward_model_logic(self, alias_inputs_local_ids, A_local_adj, items_local_ids, mask_for_seq, is_train=True):
        # 1. Get enriched item embeddings (possibly from global GCN)
        # enriched_all_item_embeddings are typically float32 if from GCN, or self.embedding.weight.dtype
        enriched_all_item_embeddings = self._get_enriched_item_embeddings()

        # 2. Process local session graph using GraphSAGE
        # hidden_session_items_processed will have the dtype of enriched_all_item_embeddings or result of GraphSAGE
        hidden_session_items_processed = self._process_session_graph_local(
            items_local_ids, A_local_adj, enriched_all_item_embeddings
        )
        
        # 3. Gather GNN outputs according to original sequence order
        # alias_inputs_local_ids maps original sequence positions to indices in items_local_ids (unique items)
        # Ensure hidden_size matches the embedding dimension
        seq_hidden_gnn_output = torch.gather(
            hidden_session_items_processed, # (B, max_unique_nodes, H)
            dim=1,
            index=alias_inputs_local_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size) # (B, max_seq_len, H)
        )
        
        # 4. Add positional encoding
        seq_hidden_with_pos = self.pos_encoder(seq_hidden_gnn_output) 
        
        # 5. Pass through Transformer encoder
        src_key_padding_mask = (mask_for_seq == 0) # True for padding positions
        output_transformer = self.transformer_encoder(
            src=seq_hidden_with_pos,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 6. Compute final scores for recommendation
        # Ensure embeddings used for scoring are of a compatible dtype with transformer output
        scores = self.compute_scores(output_transformer, mask_for_seq, enriched_all_item_embeddings)

        # 7. SSL Loss (if training)
        ssl_loss_value = torch.tensor(0.0, device=scores.device)
        if is_train and self.ssl_weight > 0:
            try:
                # Use GNN output of the last valid item in each sequence for SSL
                last_idx_for_ssl = torch.clamp(mask_for_seq.sum(1) - 1, min=0).long()
                batch_indices_ssl = torch.arange(mask_for_seq.size(0), device=seq_hidden_gnn_output.device)
                
                # Ensure indices are on the correct device
                last_idx_for_ssl = last_idx_for_ssl.to(seq_hidden_gnn_output.device) 
                
                ssl_base_emb_seq = seq_hidden_gnn_output[batch_indices_ssl, last_idx_for_ssl]
                
                ssl_emb1 = F.dropout(ssl_base_emb_seq, p=self.ssl_dropout_rate, training=True)
                ssl_emb2 = F.dropout(ssl_base_emb_seq, p=self.ssl_dropout_rate, training=True)
                ssl_loss_value = self.calculate_ssl_loss(ssl_emb1, ssl_emb2, self.ssl_temp)
            except Exception as e:
                print(f"SSL calculation error: {e}. SSL loss will be 0.")
        return scores, ssl_loss_value

    def calculate_ssl_loss(self, emb1, emb2, temperature):
        # emb1, emb2: (batch_size, hidden_size)
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        # Similarity matrix between augmented views
        sim_matrix_12 = torch.matmul(emb1_norm, emb2_norm.t()) / temperature # (B, B)
        # Numerator for InfoNCE: similarity of positive pairs (diagonal elements)
        # Denominator: sum of similarities to all other samples in batch (for each row)
        log_softmax_12 = F.log_softmax(sim_matrix_12, dim=1)
        loss_12 = -torch.diag(log_softmax_12) # InfoNCE loss for (emb1 positive to emb2)
        
        # Symmetric loss
        sim_matrix_21 = torch.matmul(emb2_norm, emb1_norm.t()) / temperature # (B, B)
        log_softmax_21 = F.log_softmax(sim_matrix_21, dim=1)
        loss_21 = -torch.diag(log_softmax_21) # InfoNCE loss for (emb2 positive to emb1)
        
        return (loss_12.mean() + loss_21.mean()) / 2.0

# --- Forward pass and Training/Evaluation utilities ---

def forward(model: SessionGraph, i_batch_indices, data_loader: utils.Data, is_train=True):
    # Fetches a batch of data and passes it through the model's main logic
    alias_inputs_np, A_local_np, items_local_np, mask_seq_np, targets_np = data_loader.get_slice(i_batch_indices)
    current_device = next(model.parameters()).device
    
    alias_inputs = torch.from_numpy(alias_inputs_np).long().to(current_device)
    # A_local_adj is now (B, N_unique, N_unique) from utils.py, suitable for GraphSAGE
    A_local_adj = torch.from_numpy(A_local_np).float().to(current_device) 
    items_local_ids = torch.from_numpy(items_local_np).long().to(current_device)
    # mask_for_seq is float in original code, but bool is often used for masks.
    # If it's float (0.0 or 1.0), it works with PyTorch masking ops.
    mask_for_seq = torch.from_numpy(mask_seq_np).float().to(current_device) 
    targets = torch.from_numpy(targets_np).long().to(current_device)

    scores, ssl_loss = model.forward_model_logic(
        alias_inputs, A_local_adj, items_local_ids, mask_for_seq, is_train=is_train
    )
    return targets, scores, ssl_loss


def evaluate_model_on_set(model: SessionGraph, eval_data: utils.Data, opt: argparse.Namespace, device: torch.device):
    """Evaluates the model on the given evaluation dataset."""
    model.eval() 
    k_metric = opt.k_metric
    recall_at_k_list, mrr_at_k_list = [], []
    
    # Determine if AMP should be used for evaluation (consistent with training if possible)
    use_amp_eval = torch.cuda.is_available() and device.type == 'cuda'


    if eval_data is not None and eval_data.length > 0:
        eval_batch_slices = eval_data.generate_batch(opt.batchSize)
        if not eval_batch_slices:
            print("Warning: No batches generated for evaluation data in evaluate_model_on_set.")
            return 0.0, 0.0, 0.0 # Return zero for metrics

        with torch.no_grad(): # Disable gradient calculations for evaluation
            for batch_indices_eval in eval_batch_slices:
                with autocast(enabled=use_amp_eval): # Enable AMP for evaluation if configured
                    targets_eval, scores_eval, _ = forward(model, batch_indices_eval, eval_data, is_train=False)

                targets_eval_cpu = targets_eval.cpu().numpy()
                # scores_eval are logits for items [1, ..., n_node-1]
                # topk returns 0-indexed indices relative to scores_eval
                _, top_k_indices_0_based = scores_eval.topk(k_metric, dim=1) 
                # Convert 0-based indices to 1-based item IDs
                top_k_item_ids = top_k_indices_0_based.cpu() + 1 
                top_k_item_ids_np = top_k_item_ids.numpy()


                for i in range(targets_eval_cpu.shape[0]): # For each sample in the batch
                    target_item_id = targets_eval_cpu[i]
                    predicted_k_item_ids = top_k_item_ids_np[i]

                    # Ensure target_item_id is a valid item ID (not padding 0, and within n_node range)
                    if target_item_id > 0 and target_item_id < model.n_node: 
                        if target_item_id in predicted_k_item_ids:
                            recall_at_k_list.append(1)
                            # Find rank of the target item among predictions
                            rank_list = np.where(predicted_k_item_ids == target_item_id)[0]
                            if len(rank_list) > 0:
                                rank = rank_list[0] + 1 # +1 for 1-based rank
                                mrr_at_k_list.append(1.0 / rank)
                            else: # Should not happen if target_item_id in predicted_k_item_ids
                                mrr_at_k_list.append(0.0)
                        else:
                            recall_at_k_list.append(0)
                            mrr_at_k_list.append(0.0)
                    # If target_item_id is padding (0) or invalid, it's not counted in metrics.
    else:
        print("Evaluation data is empty or None in evaluate_model_on_set.")
        return 0.0, 0.0, 0.0 

    final_recall_at_k = np.mean(recall_at_k_list) * 100 if recall_at_k_list else 0.0
    final_mrr_at_k = np.mean(mrr_at_k_list) * 100 if mrr_at_k_list else 0.0
    
    # Precision@K is not currently calculated by this function, returning 0.0 as before
    return final_recall_at_k, final_mrr_at_k, 0.0 


def train_test(model: SessionGraph, train_data: utils.Data, eval_data: utils.Data, opt: argparse.Namespace):
    # Determine if AMP should be used for training
    use_amp = torch.cuda.is_available() and next(model.parameters()).device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp) # GradScaler for mixed precision

    model.train() # Set model to training mode
    total_loss_epoch = 0.0
    total_rec_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0
    train_batch_slices = train_data.generate_batch(opt.batchSize)
    num_train_batches = len(train_batch_slices)

    if num_train_batches > 0:
        for step, batch_indices in enumerate(train_batch_slices):
            if not hasattr(model, 'optimizer'): 
                print("CRITICAL ERROR: model.optimizer is not set before training loop!")
                return 0.0, 0.0, 0.0 # Or raise an exception
            model.optimizer.zero_grad(set_to_none=True) # More efficient than zero_grad()

            with autocast(enabled=use_amp): # Automatic Mixed Precision context
                targets, scores, ssl_loss = forward(model, batch_indices, train_data, is_train=True)
                
                # Calculate recommendation loss only for valid targets
                valid_targets_mask = (targets > 0) & (targets < model.n_node)
                rec_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype) # Ensure same device and dtype
                
                if valid_targets_mask.any():
                    # Targets for CrossEntropyLoss should be 0-indexed (0 to C-1)
                    # Scores are for items [1, ..., n_node-1], so their indices are [0, ..., n_node-2]
                    # Target values (item IDs) need to be mapped to this 0-indexed range.
                    target_values_0_based = (targets[valid_targets_mask] - 1).clamp(min=0, max=scores.size(1) - 1)
                    rec_loss = model.loss_function(scores[valid_targets_mask], target_values_0_based)
                
                current_batch_loss = rec_loss + model.ssl_weight * ssl_loss

            # Backpropagation with GradScaler if AMP is enabled
            if use_amp:
                scaler.scale(current_batch_loss).backward()
                scaler.step(model.optimizer)
                scaler.update()
            else: # Standard backpropagation
                current_batch_loss.backward()
                model.optimizer.step()

            total_loss_epoch += current_batch_loss.item()
            # Ensure rec_loss and ssl_loss are scalars before adding to epoch totals
            total_rec_loss_epoch += rec_loss.item() if isinstance(rec_loss, torch.Tensor) else float(rec_loss) 
            total_ssl_loss_epoch += ssl_loss.item() if isinstance(ssl_loss, torch.Tensor) else float(ssl_loss)

            # Logging progress
            if (step + 1) % max(1, num_train_batches // 5) == 0 or step == num_train_batches - 1:
                avg_total_loss = total_loss_epoch / (step + 1)
                avg_rec_loss = total_rec_loss_epoch / (step + 1)
                avg_ssl_loss = total_ssl_loss_epoch / (step + 1)
                print(f'  Training Batch [{step + 1}/{num_train_batches}] '
                      f'Avg Total Loss: {avg_total_loss:.4f}, Avg Rec Loss: {avg_rec_loss:.4f}, Avg SSL Loss: {avg_ssl_loss:.4f}')
    
    # Step the learning rate scheduler after each epoch
    if hasattr(model, 'scheduler'): 
        model.scheduler.step()

    # --- Evaluation phase after epoch training ---
    model.eval() # Switch model to evaluation mode
    k_metric = opt.k_metric
    final_recall_at_k, final_mrr_at_k, final_precision_at_k = 0.0, 0.0, 0.0 # Initialize metrics

    # Perform evaluation on eval_data (validation or test set used during training)
    if eval_data is not None and eval_data.length > 0:
        current_device = next(model.parameters()).device 
        recall_eval, mrr_eval, precision_eval = evaluate_model_on_set(model, eval_data, opt, current_device)
        final_recall_at_k = recall_eval
        final_mrr_at_k = mrr_eval
        # final_precision_at_k = precision_eval # Precision is currently 0.0 from evaluate_model_on_set
        eval_set_name = "Validation" if opt.validation else "Test (during training)"
        print(f'Evaluation Results on {eval_set_name} @{k_metric}: Recall: {final_recall_at_k:.4f}%, MRR: {final_mrr_at_k:.4f}%')
    else:
        print("No evaluation data provided or evaluation data is empty during training. Skipping evaluation metrics calculation for this epoch.")

    return final_recall_at_k, final_mrr_at_k, final_precision_at_k