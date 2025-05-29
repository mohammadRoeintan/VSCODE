# model.py

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import copy
from torch.cuda.amp import autocast, GradScaler # GradScaler managed in main
# import pytz # Not used in the provided snippet
import utils
import argparse 


# IR_TIMEZONE = pytz.timezone('Asia/Tehran') # Not used

class GlobalGCN(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GlobalGCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj_sparse_matrix_normalized):
        # x should be float32
        # adj_sparse_matrix_normalized should be float32 and coalesced
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
        # x is (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :] # Add positional encoding to sequence
        return self.dropout(x)

class GNN(Module): # Local GNN for session graphs
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
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        if self.w_ih.dim() > 1: nn.init.xavier_uniform_(self.w_ih)
        if self.w_hh.dim() > 1: nn.init.xavier_uniform_(self.w_hh)
        
        # Initialize biases
        if hasattr(self, 'b_ih') and self.b_ih is not None: nn.init.uniform_(self.b_ih, -stdv, stdv)
        if hasattr(self, 'b_hh') and self.b_hh is not None: nn.init.uniform_(self.b_hh, -stdv, stdv)
        if hasattr(self, 'b_iah') and self.b_iah is not None: nn.init.uniform_(self.b_iah, -stdv, stdv)
        if hasattr(self, 'b_oah') and self.b_oah is not None: nn.init.uniform_(self.b_oah, -stdv, stdv)

        # Initialize linear layer parameters for edges
        if hasattr(self.linear_edge_in, 'weight'): nn.init.xavier_uniform_(self.linear_edge_in.weight)
        if hasattr(self.linear_edge_in, 'bias') and self.linear_edge_in.bias is not None: nn.init.zeros_(self.linear_edge_in.bias)
        if hasattr(self.linear_edge_out, 'weight'): nn.init.xavier_uniform_(self.linear_edge_out.weight)
        if hasattr(self.linear_edge_out, 'bias') and self.linear_edge_out.bias is not None: nn.init.zeros_(self.linear_edge_out.bias)


    def GNNCell(self, A, hidden):
        # A: (batch_size, num_unique_nodes_in_batch, 2 * num_unique_nodes_in_batch)
        # hidden: (batch_size, num_unique_nodes_in_batch, hidden_size)
        A_in = A[:, :, :A.size(1)]  # (batch, N_max_unique, N_max_unique)
        A_out = A[:, :, A.size(1): 2 * A.size(1)] # (batch, N_max_unique, N_max_unique)
        
        input_in = torch.matmul(A_in, self.linear_edge_in(hidden)) + self.b_iah 
        input_out = torch.matmul(A_out, self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2) # (batch, N_max_unique, 2 * hidden_size)
        
        gi = F.linear(inputs, self.w_ih, self.b_ih) # (batch, N_max_unique, gate_size)
        gh = F.linear(hidden, self.w_hh, self.b_hh) # (batch, N_max_unique, gate_size)
        
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for _ in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class TargetAwareEncoderLayer(Module): # Standard Transformer Encoder Layer
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super(TargetAwareEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) # Dropout for FFN
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) # Dropout after MHA
        self.dropout2 = nn.Dropout(dropout) # Dropout after FFN
        self.activation = activation

    def forward(self, src, src_key_padding_mask=None):
        # src: (batch_size, seq_len, d_model)
        # src_key_padding_mask: (batch_size, seq_len) - True where padded
        
        src_norm = self.norm1(src)
        sa_output, _ = self.self_attn(src_norm, src_norm, src_norm,
                                    attn_mask=None, 
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(sa_output) # Add & Norm (norm1 is pre-norm)
        
        src_norm = self.norm2(src) # Pre-norm for FFN
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_output) # Add & Norm (norm2 is pre-norm)
        return src

class TargetAwareTransformerEncoder(Module): # Stack of Transformer Encoder Layers
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
            if not current_adj_for_buffer.is_sparse:
                current_adj_for_buffer = current_adj_for_buffer.to_sparse_coo()

            if not current_adj_for_buffer.is_coalesced():
                current_adj_for_buffer = current_adj_for_buffer.coalesce()

            if current_adj_for_buffer.values().dtype != torch.float32:
                current_adj_for_buffer = torch.sparse_coo_tensor(
                    current_adj_for_buffer.indices(),
                    current_adj_for_buffer.values().float(),
                    current_adj_for_buffer.size(),
                    dtype=torch.float32,
                    device=current_adj_for_buffer.device
                ).coalesce()

            self.register_buffer('global_adj_sparse_matrix_normalized', current_adj_for_buffer)

            self.global_gcn_layers_module = nn.ModuleList()
            for _ in range(self.num_global_gcn_layers_config):
                self.global_gcn_layers_module.append(GlobalGCN(self.hidden_size, self.hidden_size))
            self.use_global_graph = True
            print(f"SessionGraph: Using {self.num_global_gcn_layers_config} global GCN layers. Sparse adj matrix shape: {self.global_adj_sparse_matrix_normalized.shape}, nnz: {self.global_adj_sparse_matrix_normalized._nnz()}, values_dtype: {self.global_adj_sparse_matrix_normalized.values().dtype}")
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
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = None 
        self.scheduler = None 
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            
            if 'embedding.weight' == name:
                 nn.init.normal_(param, mean=0, std=0.01)
                 if self.embedding.padding_idx is not None:
                     with torch.no_grad():
                         param[self.embedding.padding_idx].fill_(0)
            elif 'w_ih' in name or 'w_hh' in name or \
                 (param.dim() > 1 and 'weight' in name and 'global_gcn_layers_module' not in name and 'gnn_local' not in name): # Exclude gnn_local parameters handled by its own reset
                nn.init.xavier_uniform_(param)
            elif ('bias' in name or param.dim() == 1) and 'gnn_local' not in name: # Exclude gnn_local parameters
                nn.init.uniform_(param, -stdv, stdv)
        
        # Call reset_parameters for submodules that have it
        if hasattr(self.gnn_local, 'reset_parameters'):
            self.gnn_local.reset_parameters()

        if self.global_gcn_layers_module is not None:
            for gcn_layer in self.global_gcn_layers_module:
                 if hasattr(gcn_layer, 'linear') and hasattr(gcn_layer.linear, 'weight'):
                      nn.init.xavier_uniform_(gcn_layer.linear.weight)
                      if gcn_layer.linear.bias is not None:
                           nn.init.zeros_(gcn_layer.linear.bias)
        
        # Initialize specific linear layers if not covered by general rules
        for layer in [self.linear_one, self.linear_two, self.linear_three, self.linear_transform, self.linear_t]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)


    def _get_enriched_item_embeddings(self):
        all_item_initial_embeddings = self.embedding.weight

        if self.use_global_graph and self.global_gcn_layers_module is not None:
            with autocast(enabled=False): 
                current_embeddings_float32 = all_item_initial_embeddings.float()
                adj_float32_coalesced = self.global_adj_sparse_matrix_normalized
                
                for gcn_layer in self.global_gcn_layers_module:
                    current_embeddings_float32 = gcn_layer(current_embeddings_float32, adj_float32_coalesced)
                    current_embeddings_float32 = F.relu(current_embeddings_float32)
            return current_embeddings_float32
        else:
            return all_item_initial_embeddings


    def _process_session_graph_local(self, items_local_session_ids, A_local_session_adj, enriched_all_item_embeddings):
        hidden_local_session_enriched = F.embedding(
            items_local_session_ids,
            enriched_all_item_embeddings,
            padding_idx=0 
        ).to(A_local_session_adj.dtype) # Ensure dtype consistency for GNN
        
        hidden_local_session_processed = self.gnn_local(A_local_session_adj, hidden_local_session_enriched)
        return hidden_local_session_processed

    def _get_session_rep_v1_local_transformer(self, hidden_transformer_output, mask_for_seq):
        """ Derives session representation from Transformer outputs (View 1 for SSL). """
        # hidden_transformer_output: (batch, seq_len, hidden_size)
        # mask_for_seq: (batch, seq_len) - boolean mask, True for valid items
        
        mask_float_unsqueezed = mask_for_seq.float().unsqueeze(-1) # (batch, seq_len, 1)
        
        batch_indices = torch.arange(mask_for_seq.size(0), device=hidden_transformer_output.device)
        # Get last valid item's index for each session in batch
        last_item_indices = torch.clamp(mask_for_seq.sum(1) - 1, min=0).long()
        ht_last_item = hidden_transformer_output[batch_indices, last_item_indices] # (batch, hidden_size)

        q1 = self.linear_one(ht_last_item).unsqueeze(1) # (batch, 1, hidden_size)
        q2 = self.linear_two(hidden_transformer_output) # (batch, seq_len, hidden_size)
        
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1) # (batch, seq_len)
        # Mask out padded items before softmax
        alpha_logits_masked = alpha_logits.masked_fill(mask_for_seq == 0, torch.finfo(alpha_logits.dtype).min)
        alpha = torch.softmax(alpha_logits_masked, dim=1).unsqueeze(-1) # (batch, seq_len, 1)
        
        session_rep_v1 = (alpha * hidden_transformer_output * mask_float_unsqueezed).sum(1) # (batch, hidden_size)
        return session_rep_v1, ht_last_item


    def compute_scores(self, hidden_transformer_output, mask_for_seq, all_item_embeddings_for_scoring, session_rep_v1_local, ht_last_item):
        # session_rep_v1_local is 'a', ht_last_item is 'ht'
        mask_float_unsqueezed = mask_for_seq.float().unsqueeze(-1)
        candidate_embeds = all_item_embeddings_for_scoring[1:].to(hidden_transformer_output.dtype) # Ensure dtype match

        if self.nonhybrid:
            combined_session_rep = self.linear_transform(torch.cat([session_rep_v1_local, ht_last_item], dim=1))
            scores = torch.matmul(combined_session_rep, candidate_embeds.t())
        else:
            qt = self.linear_t(hidden_transformer_output) 
            beta_logits = torch.matmul(candidate_embeds, qt.transpose(1, 2))
            beta_logits_masked = beta_logits.masked_fill(mask_for_seq.unsqueeze(1) == 0, torch.finfo(beta_logits.dtype).min)
            beta = torch.softmax(beta_logits_masked, dim=-1)
            target_ctx = torch.matmul(beta, qt * mask_float_unsqueezed)
            final_representation = session_rep_v1_local.unsqueeze(1) + target_ctx
            scores = torch.sum(final_representation * candidate_embeds.unsqueeze(0), dim=-1)
        return scores

    def forward_model_logic(self, alias_inputs_local_ids, A_local_adj, items_local_unique_ids, mask_for_seq, is_train=True):
        enriched_all_item_embeddings = self._get_enriched_item_embeddings()

        hidden_session_items_processed = self._process_session_graph_local(
            items_local_unique_ids, A_local_adj, # A_local_adj already float from Data
            enriched_all_item_embeddings
        )

        seq_hidden_gnn_output = torch.gather(
            hidden_session_items_processed,
            dim=1,
            index=alias_inputs_local_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        ).to(enriched_all_item_embeddings.dtype) # Ensure dtype for pos_encoder

        seq_hidden_with_pos = self.pos_encoder(seq_hidden_gnn_output)
        src_key_padding_mask = (mask_for_seq == 0)
        output_transformer = self.transformer_encoder(
            src=seq_hidden_with_pos,
            src_key_padding_mask=src_key_padding_mask
        )

        session_rep_v1_local, ht_last_item = self._get_session_rep_v1_local_transformer(output_transformer, mask_for_seq)
        scores = self.compute_scores(output_transformer, mask_for_seq, enriched_all_item_embeddings, session_rep_v1_local, ht_last_item)

        ssl_loss_value = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        if is_train and self.ssl_weight > 0:
            # View 1: session_rep_v1_local (attentive sum of Transformer outputs)
            
            # View 2: Average pooling of GNN outputs (before Positional Encoding and Transformer)
            # This represents the session based on locally aggregated item features (plus global enrichment if active)
            s_v2_numerator = (seq_hidden_gnn_output * mask_for_seq.unsqueeze(-1).to(seq_hidden_gnn_output.dtype)).sum(dim=1)
            s_v2_denominator = mask_for_seq.sum(dim=1, keepdim=True).to(seq_hidden_gnn_output.dtype).clamp(min=1e-9)
            s_v2 = s_v2_numerator / s_v2_denominator

            s_v1_aug = session_rep_v1_local
            s_v2_aug = s_v2
            if self.ssl_dropout_rate > 0: # Apply SSL-specific dropout if enabled
                s_v1_aug = F.dropout(s_v1_aug, p=self.ssl_dropout_rate, training=True) # training=True for SSL dropout
                s_v2_aug = F.dropout(s_v2_aug, p=self.ssl_dropout_rate, training=True)
            
            ssl_loss_value = self.calculate_ssl_loss(s_v1_aug, s_v2_aug, self.ssl_temp)
        
        return scores, ssl_loss_value

    def calculate_ssl_loss(self, emb1, emb2, temperature):
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        sim_matrix = torch.matmul(emb1_norm, emb2_norm.t()) / temperature
        
        log_softmax_12 = F.log_softmax(sim_matrix, dim=1)
        loss_12 = -torch.diag(log_softmax_12)
        
        log_softmax_21 = F.log_softmax(sim_matrix.t(), dim=1)
        loss_21 = -torch.diag(log_softmax_21)
        
        ssl_loss = (loss_12.mean() + loss_21.mean()) / 2.0
        return ssl_loss

# --- forward function (helper for train_test and evaluate) ---
def forward(model: SessionGraph, i_batch_indices, data_loader: utils.Data, is_train=True):
    alias_inputs_np, A_local_np, items_local_unique_np, mask_seq_np, targets_np = data_loader.get_slice(i_batch_indices)
    
    current_device = next(model.parameters()).device
    
    alias_inputs = torch.from_numpy(alias_inputs_np).long().to(current_device)
    A_local_adj = torch.from_numpy(A_local_np).float().to(current_device) 
    items_local_unique_ids = torch.from_numpy(items_local_unique_np).long().to(current_device)
    mask_for_seq = torch.from_numpy(mask_seq_np).bool().to(current_device) # mask_seq_np is float, convert to bool
    targets = torch.from_numpy(targets_np).long().to(current_device)

    scores, ssl_loss = model.forward_model_logic(
        alias_inputs, A_local_adj, items_local_unique_ids, mask_for_seq, 
        is_train=is_train
    )
    return targets, scores, ssl_loss


def evaluate_model_on_set(model: SessionGraph, eval_data: utils.Data, opt: argparse.Namespace, device: torch.device):
    model.eval() 
    k_metric = opt.k_metric
    recall_at_k_list, mrr_at_k_list = [], []
    
    # Use AMP for evaluation if it was used for training, for consistency.
    # However, since gradients are not computed, its primary benefit (speed/memory for grads) isn't there.
    # Can be enabled or disabled. Let's assume consistency with training.
    use_amp_eval = torch.cuda.is_available() and device.type == 'cuda'

    if eval_data is not None and eval_data.length > 0:
        eval_batch_slices = eval_data.generate_batch(opt.batchSize)
        if not eval_batch_slices:
            print("Warning: No batches generated for evaluation data in evaluate_model_on_set.")
            return 0.0, 0.0, 0.0 

        with torch.no_grad(): 
            for batch_indices_eval in eval_batch_slices:
                with autocast(enabled=use_amp_eval): 
                    targets_eval, scores_eval, _ = forward(model, batch_indices_eval, eval_data, is_train=False)

                targets_eval_cpu = targets_eval.cpu().numpy()
                _, top_k_indices_0_based = scores_eval.topk(k_metric, dim=1)
                top_k_item_ids_1_based = top_k_indices_0_based.cpu() + 1 
                top_k_item_ids_np = top_k_item_ids_1_based.numpy()

                for i in range(targets_eval_cpu.shape[0]):
                    target_item_id = targets_eval_cpu[i]
                    if target_item_id <= 0 or target_item_id >= model.n_node: 
                        continue
                    predicted_k_item_ids = top_k_item_ids_np[i]

                    if target_item_id in predicted_k_item_ids:
                        recall_at_k_list.append(1.0)
                        rank = np.where(predicted_k_item_ids == target_item_id)[0][0] + 1
                        mrr_at_k_list.append(1.0 / rank)
                    else:
                        recall_at_k_list.append(0.0)
                        mrr_at_k_list.append(0.0)
    else:
        # print("Evaluation data is empty or None in evaluate_model_on_set.") # Can be verbose
        return 0.0, 0.0, 0.0

    final_recall_at_k = np.mean(recall_at_k_list) * 100.0 if recall_at_k_list else 0.0
    final_mrr_at_k = np.mean(mrr_at_k_list) * 100.0 if mrr_at_k_list else 0.0
    
    return final_recall_at_k, final_mrr_at_k, 0.0 # Precision not calculated


def train_test(model: SessionGraph, train_data: utils.Data, eval_data: utils.Data, opt: argparse.Namespace):
    use_amp = torch.cuda.is_available() and next(model.parameters()).device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    if not hasattr(model, 'optimizer') or model.optimizer is None: 
        print("CRITICAL ERROR: model.optimizer is not set!")
        # Fallback or raise error
        # model.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2) # Example fallback
        return 0.0, 0.0, 0.0 
    
    model.train()
    total_loss_epoch = 0.0
    total_rec_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0
    
    train_batch_slices = train_data.generate_batch(opt.batchSize)
    num_train_batches = len(train_batch_slices)

    if num_train_batches == 0:
        print("Warning: No batches generated for training data. Skipping epoch.")
        if eval_data is not None and eval_data.length > 0:
            current_device = next(model.parameters()).device
            return evaluate_model_on_set(model, eval_data, opt, current_device)
        return 0.0, 0.0, 0.0


    for step, batch_indices in enumerate(train_batch_slices):
        model.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            targets, scores, ssl_loss = forward(model, batch_indices, train_data, is_train=True)
            
            valid_targets_mask = (targets > 0) & (targets < model.n_node)
            rec_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype) 

            if valid_targets_mask.any():
                target_values_0_based = (targets[valid_targets_mask] - 1).clamp(min=0, max=scores.size(1)-1)
                rec_loss = model.loss_function(scores[valid_targets_mask], target_values_0_based)
            
            current_batch_loss = rec_loss + model.ssl_weight * ssl_loss

        if use_amp:
            scaler.scale(current_batch_loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            current_batch_loss.backward()
            model.optimizer.step()

        total_loss_epoch += current_batch_loss.item()
        total_rec_loss_epoch += rec_loss.item() 
        total_ssl_loss_epoch += ssl_loss.item()

        if (step + 1) % max(1, num_train_batches // 5) == 0 or step == num_train_batches - 1:
            avg_total_loss = total_loss_epoch / (step + 1)
            avg_rec_loss = total_rec_loss_epoch / (step + 1)
            avg_ssl_loss = total_ssl_loss_epoch / (step + 1)
            print(f'  Training Batch [{step + 1}/{num_train_batches}] '
                  f'Avg Total Loss: {avg_total_loss:.4f}, Avg Rec Loss: {avg_rec_loss:.4f}, Avg SSL Loss: {avg_ssl_loss:.4f}')

    if hasattr(model, 'scheduler') and model.scheduler is not None:
        model.scheduler.step()

    model.eval() 
    final_recall_at_k, final_mrr_at_k, final_precision_at_k = 0.0, 0.0, 0.0

    if eval_data is not None and eval_data.length > 0:
        current_device = next(model.parameters()).device
        recall_eval, mrr_eval, precision_eval = evaluate_model_on_set(model, eval_data, opt, current_device)
        final_recall_at_k = recall_eval
        final_mrr_at_k = mrr_eval
        # final_precision_at_k = precision_eval 
        eval_set_name = "Validation" if opt.validation else "Test (during training)"
        print(f'Epoch End Evaluation @{opt.k_metric} on {eval_set_name}: Recall: {final_recall_at_k:.4f}%, MRR: {final_mrr_at_k:.4f}%')
    else:
        print("No evaluation data (or empty) for this epoch's evaluation.")

    return final_recall_at_k, final_mrr_at_k, final_precision_at_k