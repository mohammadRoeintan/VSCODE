# model.py

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import copy
from torch.cuda.amp import autocast, GradScaler
import pytz

IR_TIMEZONE = pytz.timezone('Asia/Tehran')


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
        # This operation might produce a 'Half' tensor for 'support' under autocast
        support = self.linear(x)  # Potentially (N, out_features) in Half

        # --- Ensure inputs to torch.sparse.mm are float32 ---
        current_adj = adj_sparse_matrix_normalized
        current_support = support

        # 1. Ensure the dense matrix ('support') is float32
        if current_support.dtype == torch.half:
            current_support = current_support.float()

        # 2. Ensure the sparse matrix ('adj_sparse_matrix_normalized') is float32.
        #    Sparse tensors store their values in a dense tensor, check its dtype.
        if current_adj.is_sparse:
            if current_adj.values().dtype == torch.half:
                # Re-create the sparse tensor with float32 values if its values are half
                current_adj = torch.sparse_coo_tensor(
                    current_adj.indices(),
                    current_adj.values().float(), # Cast values to float32
                    current_adj.size(),
                    dtype=torch.float32, # Explicitly set dtype for the new sparse tensor
                    device=current_adj.device
                )
        elif current_adj.dtype == torch.half: # Should not happen if it's always sparse
             current_adj = current_adj.float()
        # --- End of float32 enforcement ---
        
        # Perform sparse matrix multiplication with float32 tensors
        output = torch.sparse.mm(current_adj, current_support) # A_hat * X * W
        
        return output


# PositionalEncoding (بدون تغییر نسبت به نسخه قبلی که با batch_first=True سازگار شد)
class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model) -> (max_len, d_model) if not unsqueezed before transpose
        self.register_buffer('pe', pe) # pe: (max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe: (max_len, d_model)
        # نیاز داریم self.pe[:x.size(1), :] را به x اضافه کنیم
        # self.pe[:x.size(1), :] shape: (seq_len, d_model)
        # باید با (batch_size, seq_len, d_model) جمع شود، پس pe باید unsqueeze شود یا x transpose شود
        # x = x + self.pe[:x.size(1), :].unsqueeze(0) # unsqueeze(0) to make it (1, seq_len, d_model) for broadcasting
        x = x + self.pe[:x.size(1), :] # Broadcasting کار می‌کند اگر x (B,S,D) و pe (S,D) باشد
        return self.dropout(x)


# GNN (برای گراف محلی - بدون تغییر)
class GNN(Module):
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
        for weight in self.parameters():
            if weight.dim() > 1: nn.init.xavier_uniform_(weight)
            else: nn.init.uniform_(weight, -stdv, stdv)

    def GNNCell(self, A, hidden):
        A_in = A[:, :, :A.shape[1]]
        A_out = A[:, :, A.shape[1]: 2 * A.shape[1]]
        input_in = torch.matmul(A_in, self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A_out, self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
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


# TargetAwareEncoderLayer (batch_first=True)
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

    def forward(self, src, src_key_padding_mask=None): # candidate_embeddings_global حذف شد
        # src: (Batch, Seq, Feature)
        src2 = self.norm1(src)
        sa_output, _ = self.self_attn(src2, src2, src2,
                                    attn_mask=None, # انکودر استاندارد معمولا ماسک توجه ندارد
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(sa_output)
        src2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(ff_output)
        return src


# TargetAwareTransformerEncoder
class TargetAwareTransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TargetAwareTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None): # candidate_embeddings_global حذف شد
        output = src
        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_key_padding_mask) # mask هم معمولا برای انکودر None است
        if self.norm is not None:
            output = self.norm(output)
        return output


# SessionGraph
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
        self.global_gcn_layers_module = None # تغییر نام برای وضوح
        if self.num_global_gcn_layers_config > 0 and global_adj_sparse_matrix is not None:
            # ثبت ماتریس پراکنده گلوبال به عنوان بافر
            # اطمینان از اینکه ماتریس پراکنده است
            if not global_adj_sparse_matrix.is_sparse:
                print("Warning: Global adjacency matrix is not sparse. Converting to sparse COO.")
                global_adj_sparse_matrix = global_adj_sparse_matrix.to_sparse_coo()
            self.register_buffer('global_adj_sparse_matrix_normalized', global_adj_sparse_matrix)
            
            self.global_gcn_layers_module = nn.ModuleList()
            current_dim = self.hidden_size
            for i in range(self.num_global_gcn_layers_config):
                gcn_layer = GlobalGCN(current_dim, self.hidden_size) # D_out = hidden_size
                self.global_gcn_layers_module.append(gcn_layer)
                # current_dim = self.hidden_size # اگر ابعاد ثابت بماند

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
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.loss_function = nn.CrossEntropyLoss()
        # Optimizer و Scheduler در main.py ساخته و به مدل پاس داده نمی‌شوند، بلکه مستقیما پارامترهای مدل را می‌گیرند
        # بنابراین، model.optimizer و model.scheduler را در main.py مستقیما مقداردهی می‌کنیم.
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'weight' in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name: # بایاس‌ها
                    nn.init.uniform_(param, -stdv, stdv)
                elif param.dim() == 1 and 'weight' not in name and 'bias' not in name: # سایر پارامترهای یک بعدی (مثل b_iah)
                    nn.init.uniform_(param, -stdv, stdv)
        
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].fill_(0)
        
        # مقداردهی اولیه GCN گلوبال (اگر دستی انجام شود)
        if self.global_gcn_layers_module is not None:
            for gcn_layer in self.global_gcn_layers_module:
                if hasattr(gcn_layer, 'linear') and hasattr(gcn_layer.linear, 'weight'):
                     nn.init.xavier_uniform_(gcn_layer.linear.weight)


    def _get_enriched_item_embeddings(self):
        all_item_initial_embeddings = self.embedding.weight
        
        if self.use_global_graph and self.global_gcn_layers_module is not None:
            current_embeddings = all_item_initial_embeddings
            # global_adj_sparse_matrix_normalized باید از قبل روی device مدل باشد (چون بافر است)
            adj = self.global_adj_sparse_matrix_normalized
            
            for gcn_layer in self.global_gcn_layers_module:
                current_embeddings = gcn_layer(current_embeddings, adj)
                current_embeddings = F.relu(current_embeddings) 
                # می‌توان Dropout هم اضافه کرد: current_embeddings = F.dropout(current_embeddings, p=..., training=self.training)
            
            # ترکیب با امبدینگ اولیه (Residual connection) می‌تواند مفید باشد
            # return all_item_initial_embeddings + current_embeddings
            return current_embeddings # یا فقط خروجی GCN
        else:
            return all_item_initial_embeddings


    def _process_session_graph_local(self, items_local_session_ids, A_local_session_adj, enriched_all_item_embeddings):
        hidden_local_session_enriched = F.embedding(items_local_session_ids, enriched_all_item_embeddings, padding_idx=0)
        hidden_local_session_processed = self.gnn_local(A_local_session_adj, hidden_local_session_enriched)
        return hidden_local_session_processed


    def compute_scores(self, hidden_transformer_output, mask, all_item_embeddings_for_scoring):
        mask = mask.float()
        batch_indices = torch.arange(mask.size(0), device=hidden_transformer_output.device)
        last_item_indices = torch.clamp(mask.sum(1) - 1, min=0).long().to(hidden_transformer_output.device)
        ht = hidden_transformer_output[batch_indices, last_item_indices]

        q1 = self.linear_one(ht).unsqueeze(1)
        q2 = self.linear_two(hidden_transformer_output)
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1)
        
        alpha_logits_masked = alpha_logits.masked_fill(mask == 0, torch.finfo(alpha_logits.dtype).min)
        alpha = torch.softmax(alpha_logits_masked, dim=1)
        
        a = (alpha.unsqueeze(-1) * hidden_transformer_output * mask.unsqueeze(-1)).sum(1)
        
        candidate_embeds = all_item_embeddings_for_scoring[1:] # بدون پدینگ (آیتم 0)

        if self.nonhybrid:
            combined_session_rep = self.linear_transform(torch.cat([a, ht], dim=1))
            scores = torch.matmul(combined_session_rep, candidate_embeds.t())
        else:
            qt = self.linear_t(hidden_transformer_output)
            beta_logits = torch.matmul(candidate_embeds, qt.transpose(1, 2))
            beta_logits_masked = beta_logits.masked_fill(mask.unsqueeze(1) == 0, torch.finfo(beta_logits.dtype).min)
            beta = torch.softmax(beta_logits_masked, dim=-1)
            target_ctx = torch.matmul(beta, qt * mask.unsqueeze(-1))
            final_representation = a.unsqueeze(1) + target_ctx
            scores = torch.sum(final_representation * candidate_embeds.unsqueeze(0), dim=-1)
        
        return scores


    def forward_model_logic(self, alias_inputs_local_ids, A_local_adj, items_local_ids, mask_for_seq, is_train=True):
        enriched_all_item_embeddings = self._get_enriched_item_embeddings()
        hidden_session_items_processed = self._process_session_graph_local(items_local_ids, A_local_adj, enriched_all_item_embeddings)
        
        seq_hidden_gnn_output = torch.gather(
            hidden_session_items_processed, 
            dim=1, 
            index=alias_inputs_local_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        
        seq_hidden_with_pos = self.pos_encoder(seq_hidden_gnn_output)
        
        src_key_padding_mask = (mask_for_seq == 0)
        output_transformer = self.transformer_encoder(
            src=seq_hidden_with_pos,
            src_key_padding_mask=src_key_padding_mask
        )
        
        scores = self.compute_scores(output_transformer, mask_for_seq, enriched_all_item_embeddings)

        ssl_loss_value = torch.tensor(0.0, device=scores.device)
        if is_train and self.ssl_weight > 0:
            try:
                last_idx_for_ssl = torch.clamp(mask_for_seq.sum(1) - 1, min=0).long()
                batch_indices_ssl = torch.arange(mask_for_seq.size(0), device=seq_hidden_gnn_output.device)
                last_idx_for_ssl = last_idx_for_ssl.to(seq_hidden_gnn_output.device)
                ssl_base_emb_seq = seq_hidden_gnn_output[batch_indices_ssl, last_idx_for_ssl]
                
                ssl_emb1 = F.dropout(ssl_base_emb_seq, p=self.ssl_dropout_rate, training=True)
                ssl_emb2 = F.dropout(ssl_base_emb_seq, p=self.ssl_dropout_rate, training=True)
                ssl_loss_value = self.calculate_ssl_loss(ssl_emb1, ssl_emb2, self.ssl_temp)
            except Exception as e:
                print(f"SSL calculation error: {e}")
        
        return scores, ssl_loss_value

    def calculate_ssl_loss(self, emb1, emb2, temperature): # بدون تغییر
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        sim_matrix_12 = torch.matmul(emb1, emb2.t()) / temperature
        log_softmax_12 = F.log_softmax(sim_matrix_12, dim=1)
        loss_12 = -torch.diag(log_softmax_12)
        sim_matrix_21 = torch.matmul(emb2, emb1.t()) / temperature
        log_softmax_21 = F.log_softmax(sim_matrix_21, dim=1)
        loss_21 = -torch.diag(log_softmax_21)
        return (loss_12.mean() + loss_21.mean()) / 2.0


# تابع forward wrapper (بدون تغییر نسبت به نسخه قبلی)
def forward(model: SessionGraph, i, data, is_train=True):
    alias_inputs_np, A_local_np, items_local_np, mask_seq_np, targets_np = data.get_slice(i)
    current_device = next(model.parameters()).device
    alias_inputs = torch.from_numpy(alias_inputs_np).long().to(current_device)
    A_local_adj = torch.from_numpy(A_local_np).float().to(current_device)
    items_local_ids = torch.from_numpy(items_local_np).long().to(current_device)
    mask_for_seq = torch.from_numpy(mask_seq_np).float().to(current_device)
    targets = torch.from_numpy(targets_np).long().to(current_device)
    scores, ssl_loss = model.forward_model_logic(
        alias_inputs, A_local_adj, items_local_ids, mask_for_seq, is_train=is_train
    )
    return targets, scores, ssl_loss


# تابع train_test (بدون تغییر عمده نسبت به نسخه قبلی، فقط k_metric از opt گرفته می‌شود)
def train_test(model, train_data, test_data, opt):
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp) 
    
    # Optimizer و Scheduler باید اینجا ساخته شوند چون به پارامترهای مدل نیاز دارند
    # اینها قبلا در __init__ مدل بودند، اما بهتر است در main یا اینجا باشند
    # اگر در __init__ مدل هستند، باید اطمینان حاصل شود که optimizer و scheduler مدل در اینجا استفاده می‌شوند.
    # در اینجا فرض می‌کنیم optimizer و scheduler در main.py به عنوان attribute های مدل تنظیم شده‌اند.
    # model.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    # model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)


    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_ir = now_utc.astimezone(IR_TIMEZONE)
    # epoch_num = model.scheduler.last_epoch if hasattr(model, 'scheduler') else 0 # برای نمایش بهتر
    # print(f'Start training (Epoch {epoch_num}): {now_ir.strftime("%Y-%m-%d %H:%M:%S")}')
    
    model.train()
    total_loss_epoch = 0.0
    total_rec_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0
    
    slices = train_data.generate_batch(opt.batchSize)
    num_batches = len(slices)

    if num_batches == 0:
        print("Warning: No batches to train on.")
        # برای جلوگیری از خطا در ادامه، اگر داده تست هم خالی باشد، مقادیر صفر برمی‌گردانیم
        if test_data is None or test_data.length == 0:
            return 0.0, 0.0, 0.0
        # در غیر این صورت، فقط بخش آموزش را رد می‌کنیم و به ارزیابی می‌رویم (اگرچه منطقی نیست)


    for step, batch_indices in enumerate(slices):
        # اطمینان از وجود optimizer در مدل
        if not hasattr(model, 'optimizer'):
            print("Error: model.optimizer is not set. Please create optimizer in main.py and assign to model.")
            return 0.0, 0.0, 0.0 # خطا
            
        model.optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=use_amp):
            targets, scores, ssl_loss = forward(model, batch_indices, train_data, is_train=True)
            valid_targets_mask = (targets > 0) & (targets < model.n_node)
            rec_loss = torch.tensor(0.0, device=scores.device)
            if valid_targets_mask.any():
                target_values_0_based = (targets[valid_targets_mask] - 1).clamp(0, scores.size(1) - 1)
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
        total_ssl_loss_epoch += ssl_loss.item() if isinstance(ssl_loss, torch.Tensor) else ssl_loss
        
        if (step + 1) % max(1, num_batches // 5) == 0 or step == num_batches - 1:
            avg_total_loss = total_loss_epoch / (step + 1)
            avg_rec_loss = total_rec_loss_epoch / (step + 1)
            avg_ssl_loss = total_ssl_loss_epoch / (step + 1)
            current_epoch_display = model.scheduler.last_epoch if hasattr(model, 'scheduler') else 'N/A' # برای نمایش بهتر
            print(f'Epoch [{current_epoch_display}/{opt.epoch}] Batch [{step + 1}/{num_batches}] '
                  f'Total Loss: {avg_total_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}, SSL Loss: {avg_ssl_loss:.4f}')
    
    if hasattr(model, 'scheduler'):
        model.scheduler.step()
    
    model.eval()
    k_metric = opt.k_metric # از opt خوانده می‌شود
    
    if test_data is None or test_data.length == 0:
        print("No evaluation data provided. Skipping evaluation.")
        return 0.0, 0.0, 0.0

    hit_at_k, mrr_at_k = [], []
    test_slices = test_data.generate_batch(opt.batchSize)

    if not test_slices: # اگر test_data خالی باشد، test_slices هم خالی خواهد بود
        print("No batches to evaluate on.")
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch_indices_test in test_slices:
            targets_test, scores_test, _ = forward(model, batch_indices_test, test_data, is_train=False)
            _, top_k_indices_0_based = scores_test.topk(k_metric, dim=1)
            top_k_item_ids = top_k_indices_0_based + 1
            targets_test_np = targets_test.cpu().numpy()
            top_k_item_ids_np = top_k_item_ids.cpu().numpy()

            for i in range(targets_test_np.shape[0]):
                target_item_id = targets_test_np[i]
                predicted_item_ids_at_k = top_k_item_ids_np[i]
                if target_item_id > 0 and target_item_id < model.n_node:
                    if target_item_id in predicted_item_ids_at_k:
                        hit_at_k.append(1)
                        rank = np.where(predicted_item_ids_at_k == target_item_id)[0][0] + 1
                        mrr_at_k.append(1.0 / rank)
                    else:
                        hit_at_k.append(0)
                        mrr_at_k.append(0.0)
    
    final_recall_at_k = np.mean(hit_at_k) * 100 if hit_at_k else 0.0
    final_mrr_at_k = np.mean(mrr_at_k) * 100 if mrr_at_k else 0.0
    
    print(f'Evaluation Results @{k_metric}: Recall: {final_recall_at_k:.4f}%, MRR: {final_mrr_at_k:.4f}%')
    
    return final_recall_at_k, final_mrr_at_k, 0.0 # برگرداندن صفر برای precision