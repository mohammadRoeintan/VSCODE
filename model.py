# model.py

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import copy
from torch.cuda.amp import autocast, GradScaler # برای آموزش با دقت ترکیبی
import pytz

IR_TIMEZONE = pytz.timezone('Asia/Tehran')


class GlobalGCN(Module):
    """
    یک لایه GCN ساده برای گراف گلوبال.
    A_hat * X * W
    A_hat: ماتریس همسایگی نرمال شده گلوبال (N, N)
    X: امبدینگ های ورودی آیتم ها (N, D_in)
    W: ماتریس وزن قابل یادگیری (D_in, D_out)
    خروجی: (N, D_out)
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GlobalGCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # nn.init.xavier_uniform_(self.linear.weight) # مقداردهی اولیه در reset_parameters مدل اصلی

    def forward(self, x, adj_matrix_normalized):
        # x: (N, in_features), adj_matrix_normalized: (N, N)
        support = self.linear(x)  # X * W : (N, out_features)
        # اگر adj_matrix_normalized پراکنده (sparse) باشد، از torch.sparse.mm استفاده کنید
        if adj_matrix_normalized.is_sparse:
            output = torch.sparse.mm(adj_matrix_normalized, support) # A_hat * X * W
        else:
            output = torch.matmul(adj_matrix_normalized, support) # A_hat * X * W
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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model) اگر batch_first=False در ترانسفورمر
        # x shape: (batch_size, seq_len, d_model) اگر batch_first=True در ترانسفورمر
        # کد شما از batch_first=True برای MultiheadAttention استفاده نمی‌کند، پس ورودی ترنسفورمر (seq_len, batch, feature) است
        # اما خروجی GNN شما و ورودی به PositionalEncoding (batch, seq_len, feature) است.
        # PositionalEncoding شما انتظار (seq_len, batch, feature) دارد.
        # x = x + self.pe[:x.size(0), :] # اگر x (seq_len, ...) باشد
        # اگر x (batch, seq_len, d_model) است:
        x_transposed = x.transpose(0,1) # (seq_len, batch, d_model)
        x_transposed = x_transposed + self.pe[:x_transposed.size(0), :]
        return self.dropout(x_transposed.transpose(0,1)) # برگرداندن به (batch, seq_len, d_model)


class GNN(Module): # GNN برای گراف محلی (بدون تغییر)
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
            if weight.dim() > 1:
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.uniform_(weight, -stdv, stdv)

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


class TargetAwareEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super(TargetAwareEncoderLayer, self).__init__()
        # در MultiheadAttention، batch_first=True راحت‌تر است اگر ورودی (batch, seq, feature) باشد
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src, candidate_embeddings_global, src_mask=None, src_key_padding_mask=None):
        # src: (Batch, Seq, Feature)
        # candidate_embeddings_global: (Num_candidates, Feature) - در این پیاده‌سازی فعلا استفاده نمی‌شود
        src2 = self.norm1(src)
        # MultiheadAttention با batch_first=True: query, key, value باید (N, L, E) یا (L, N, E) باشند
        # src_key_padding_mask: (N, S)
        sa_output, _ = self.self_attn(src2, src2, src2,
                                    attn_mask=src_mask, # برای دیکدر ترنسفورمر معمولا استفاده می‌شود
                                    key_padding_mask=src_key_padding_mask) # (N,S) True برای مقادیر پد شده
        src = src + self.dropout1(sa_output)
        
        src2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(ff_output)
        return src


class TargetAwareTransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TargetAwareTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm # معمولا یک LayerNorm است

    def forward(self, src, candidate_embeddings_global, mask=None, src_key_padding_mask=None):
        # src: (Batch, Seq, Feature)
        # candidate_embeddings_global: استفاده نشده در هر لایه انکودر استاندارد، ممکن است برای مکانیزم توجه خاصی باشد
        output = src
        for mod in self.layers:
            output = mod(output, candidate_embeddings_global, 
                        src_mask=mask, 
                        src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class SessionGraph(Module):
    def __init__(self, opt, n_node, global_adj_matrix=None): # global_adj_matrix اضافه شد
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.nonhybrid = opt.nonhybrid
        self.ssl_weight = opt.ssl_weight
        self.ssl_temp = opt.ssl_temp
        self.ssl_dropout_rate = opt.ssl_dropout_rate
        self.num_global_gcn_layers = opt.global_gcn_layers
        
        # Embedding اولیه آیتم‌ها
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)
        
        # --- بخش گراف گلوبال ---
        self.use_global_graph = False
        if self.num_global_gcn_layers > 0 and global_adj_matrix is not None:
            # global_adj_matrix باید روی همان دستگاهی باشد که مدل هست.
            # ثبت به عنوان بافر تا با مدل جابجا شود و در state_dict ذخیره شود.
            self.register_buffer('global_adj_matrix_normalized', global_adj_matrix.float()) # اطمینان از float بودن
            
            self.global_gcn_layers = nn.ModuleList()
            # لایه اول GCN گلوبال: hidden_size -> hidden_size
            self.global_gcn_layers.append(GlobalGCN(self.hidden_size, self.hidden_size))
            for _ in range(1, self.num_global_gcn_layers):
                self.global_gcn_layers.append(GlobalGCN(self.hidden_size, self.hidden_size))
            
            self.use_global_graph = True
            print(f"SessionGraph initialized with {self.num_global_gcn_layers} global GCN layer(s). Global adj matrix shape: {self.global_adj_matrix_normalized.shape}")
        else:
            self.global_adj_matrix_normalized = None
            self.global_gcn_layers = None
            print("SessionGraph initialized WITHOUT global graph processing.")
        # --- پایان بخش گراف گلوبال ---

        # GNN برای گراف محلی سِشِن
        self.gnn_local = GNN(self.hidden_size, step=opt.step) # تغییر نام برای وضوح
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self.hidden_size, getattr(opt, 'dropout', 0.1))
        
        # Target-Aware Transformer Encoder
        ta_encoder_layer = TargetAwareEncoderLayer(
            d_model=self.hidden_size,
            nhead=getattr(opt, 'nhead', 2),
            dim_feedforward=getattr(opt, 'ff_hidden', 256),
            dropout=getattr(opt, 'dropout', 0.1)
        )
        self.transformer_encoder = TargetAwareTransformerEncoder(
            encoder_layer=ta_encoder_layer,
            num_layers=getattr(opt, 'nlayers', 2),
            norm=nn.LayerNorm(self.hidden_size) # نرمال‌سازی نهایی بعد از لایه‌های ترانسفورمر
        )
        
        # لایه‌های خطی برای محاسبه امتیاز نهایی
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True) # برای حالت nonhybrid
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # برای حالت hybrid
        
        # تابع هزینه و بهینه‌ساز
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1: # برای ماتریس‌های وزن
                nn.init.xavier_uniform_(weight)
            else: # برای بایاس‌ها
                nn.init.uniform_(weight, -stdv, stdv)
        
        # مقداردهی اولیه وزن embedding پدینگ به صفر
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].fill_(0)
        
        # مقداردهی اولیه وزن‌های لایه‌های GCN گلوبال اگر وجود دارند
        if self.global_gcn_layers is not None:
            for gcn_layer in self.global_gcn_layers:
                if hasattr(gcn_layer, 'linear') and hasattr(gcn_layer.linear, 'weight'):
                    nn.init.xavier_uniform_(gcn_layer.linear.weight)
                    if gcn_layer.linear.bias is not None:
                         nn.init.uniform_(gcn_layer.linear.bias, -stdv, stdv)


    def _get_enriched_item_embeddings(self):
        """
        ابتدا embeddingهای اولیه آیتم‌ها را می‌گیرد، سپس اگر GCN گلوبال فعال باشد،
        آن‌ها را با عبور از لایه‌های GCN گلوبال غنی‌سازی می‌کند.
        """
        all_item_initial_embeddings = self.embedding.weight # (n_node, hidden_size)
        
        if self.use_global_graph and self.global_gcn_layers is not None:
            current_embeddings = all_item_initial_embeddings
            # اطمینان از اینکه ماتریس همسایگی گلوبال روی همان دستگاهی است که مدل هست
            # این کار باید یکبار در زمان انتقال مدل به device انجام شود اگر به عنوان بافر ثبت شده.
            # self.global_adj_matrix_normalized باید از قبل روی device صحیح باشد.
            adj = self.global_adj_matrix_normalized 
            if adj.device != current_embeddings.device: # بررسی اضافی، نباید لازم باشد اگر بافر است
                adj = adj.to(current_embeddings.device)

            for gcn_layer in self.global_gcn_layers:
                current_embeddings = gcn_layer(current_embeddings, adj)
                current_embeddings = F.relu(current_embeddings) # یا هر تابع فعال‌سازی دیگر
                # اینجا می‌توان Dropout هم اضافه کرد
            # ترکیب نهایی می‌تواند جمع با امبدینگ اولیه باشد (residual connection) یا فقط خروجی GCN
            # return all_item_initial_embeddings + current_embeddings # مثال: ترکیب با جمع
            return current_embeddings # استفاده از خروجی مستقیم GCN
        else:
            return all_item_initial_embeddings


    def _process_session_graph_local(self, items_local_session_ids, A_local_session_adj, enriched_all_item_embeddings):
        """
        پردازش گراف محلی سِشِن با استفاده از embeddingهای غنی‌شده.
        items_local_session_ids: IDهای آیتم‌های یکتا در سِشِن‌های بچ فعلی (B, max_n_node_in_batch)
        A_local_session_adj: ماتریس همسایگی گراف محلی برای بچ (B, max_n_node_in_batch, 2*max_n_node_in_batch)
        enriched_all_item_embeddings: تمام embeddingهای آیتم غنی‌شده (n_node, hidden_size)
        خروجی: نمایش آیتم‌های سِشِن پس از GNN محلی (B, max_n_node_in_batch, hidden_size)
        """
        # گرفتن embeddingهای غنی‌شده برای آیتم‌های خاص این سِشِن‌ها
        # F.embedding برای padding_idx هم کار می‌کند.
        hidden_local_session_enriched = F.embedding(items_local_session_ids, enriched_all_item_embeddings, padding_idx=0)
        
        # پردازش با GNN محلی
        hidden_local_session_processed = self.gnn_local(A_local_session_adj, hidden_local_session_enriched)
        return hidden_local_session_processed


    def compute_scores(self, hidden_transformer_output, mask, all_item_embeddings_for_scoring):
        """
        محاسبه امتیاز نهایی برای آیتم‌های کاندید.
        hidden_transformer_output: خروجی ترانسفورمر (B, L, D)
        mask: ماسک برای توالی‌ها (B, L)
        all_item_embeddings_for_scoring: تمام embeddingهای آیتم غنی‌شده (n_node, D) برای امتیازدهی
        """
        mask = mask.float() # اطمینان از نوع داده
        
        # استخراج نمایش آخرین آیتم توالی از خروجی ترانسفورمر
        batch_indices = torch.arange(mask.size(0), device=hidden_transformer_output.device)
        # اطمینان از اینکه ایندکس‌های آخرین آیتم هم روی همان دستگاه هستند و از نوع long
        last_item_indices = torch.clamp(mask.sum(1) - 1, min=0).long().to(hidden_transformer_output.device)
        ht = hidden_transformer_output[batch_indices, last_item_indices] # (B, D)

        # مکانیزم توجه برای ترکیب نمایش آیتم‌ها در توالی (local preference)
        q1 = self.linear_one(ht).unsqueeze(1) # (B, 1, D)
        q2 = self.linear_two(hidden_transformer_output) # (B, L, D)
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1) # (B, L)
        
        alpha_logits_masked = alpha_logits.masked_fill(mask == 0, torch.finfo(alpha_logits.dtype).min)
        alpha = torch.softmax(alpha_logits_masked, dim=1) # (B, L)
        
        # نمایش ترکیبی سِشِن (local preference)
        a = (alpha.unsqueeze(-1) * hidden_transformer_output * mask.unsqueeze(-1)).sum(1) # (B, D)
        
        # گرفتن embedding کاندیداها (بدون پدینگ) از embeddingهای غنی‌شده
        candidate_embeds = all_item_embeddings_for_scoring[1:]  # (n_node-1, D)
        num_candidates = candidate_embeds.size(0)

        if self.nonhybrid:
            # ترکیب نمایش سِشِن (a) و نمایش آخرین آیتم (ht)
            combined_session_rep = self.linear_transform(torch.cat([a, ht], dim=1)) # (B, D)
            # محاسبه امتیاز با ضرب داخلی بین نمایش ترکیبی سِشِن و کاندیداها
            scores = torch.matmul(combined_session_rep, candidate_embeds.t()) # (B, n_node-1)
        else: # حالت hybrid (استفاده از توجه به آیتم هدف - target attention)
            # qt: (B, L, D) - تبدیل خطی از خروجی ترانسفورمر برای توجه به هدف
            qt = self.linear_t(hidden_transformer_output) 
            
            # beta_logits: (B, num_candidates, L) - امتیاز شباهت بین هر کاندیدا و هر آیتم در توالی
            beta_logits = torch.matmul(candidate_embeds, qt.transpose(1, 2))
            
            # اعمال ماسک روی پدینگ‌ها در توالی
            # mask.unsqueeze(1) -> (B, 1, L)
            beta_logits_masked = beta_logits.masked_fill(mask.unsqueeze(1) == 0, torch.finfo(beta_logits.dtype).min)
            beta = torch.softmax(beta_logits_masked, dim=-1) # Softmax over L (sequence length) -> (B, num_candidates, L)
            
            # target_ctx: (B, num_candidates, D) - نمایش زمینه هدف برای هر کاندیدا
            # qt * mask.unsqueeze(-1) -> (B, L, D) - صفر کردن نمایش آیتم‌های پد شده
            target_ctx = torch.matmul(beta, qt * mask.unsqueeze(-1))
            
            # final_representation: (B, num_candidates, D) - ترکیب نمایش کلی سِشِن (a) با زمینه هدف
            final_representation = a.unsqueeze(1) + target_ctx # a.unsqueeze(1) -> (B, 1, D)
            
            # scores: (B, num_candidates) - امتیاز نهایی با ضرب داخلی
            scores = torch.sum(final_representation * candidate_embeds.unsqueeze(0), dim=-1)
        
        return scores

    # این تابع حالا نقش forward اصلی را ایفا می‌کند
    def forward_model_logic(self, alias_inputs_local_ids, A_local_adj, items_local_ids, mask_for_seq, is_train=True):
        # alias_inputs_local_ids: (B, L) - ایندکس‌های محلی آیتم‌ها در توالی برای gather کردن از خروجی GNN محلی
        # A_local_adj: (B, max_nodes_in_batch, 2*max_nodes_in_batch) - ماتریس همسایگی GNN محلی
        # items_local_ids: (B, max_nodes_in_batch) - IDهای آیتم‌های یکتا در هر سِشِن بچ برای GNN محلی
        # mask_for_seq: (B, L) - ماسک برای توالی‌ها
        
        # 1. غنی‌سازی تمام embeddingهای آیتم‌ها با استفاده از GCN گلوبال (اگر فعال باشد)
        enriched_all_item_embeddings = self._get_enriched_item_embeddings() # (n_node, D)

        # 2. پردازش گراف محلی سِشِن با استفاده از embeddingهای غنی‌شده
        # خروجی: (B, max_nodes_in_batch, D)
        hidden_session_items_processed = self._process_session_graph_local(items_local_ids, A_local_adj, enriched_all_item_embeddings)

        # 3. بازسازی ترتیب توالی از خروجی GNN محلی (با استفاده از alias_inputs)
        # alias_inputs_local_ids باید ایندکس‌هایی برای hidden_session_items_processed باشند.
        # hidden_session_items_processed: (B, max_nodes_in_batch, D)
        # alias_inputs_local_ids: (B, L)
        # seq_hidden_gnn_output: (B, L, D)
        seq_hidden_gnn_output = torch.gather(
            hidden_session_items_processed, 
            dim=1, 
            index=alias_inputs_local_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        
        # 4. اضافه کردن Positional Encoding
        # ورودی به pos_encoder باید (B, L, D) باشد اگر در PositionalEncoding تغییرات اعمال شده
        seq_hidden_with_pos = self.pos_encoder(seq_hidden_gnn_output) # (B, L, D)
        
        # 5. عبور از Target-Aware Transformer Encoder
        # src_key_padding_mask: (B, L), True برای مقادیر پد شده
        src_key_padding_mask = (mask_for_seq == 0) 
        # candidate_embeddings_global برای ترانسفورمر (اگر استفاده می‌شود) باید غنی‌شده باشند
        transformer_candidate_embeds = enriched_all_item_embeddings[1:] # بدون پدینگ
        
        output_transformer = self.transformer_encoder(
            src=seq_hidden_with_pos, # (B, L, D)
            candidate_embeddings_global=transformer_candidate_embeds, 
            src_key_padding_mask=src_key_padding_mask
        ) # خروجی: (B, L, D)
        
        # 6. محاسبه امتیاز نهایی
        # all_item_embeddings_for_scoring باید غنی‌شده باشند
        scores = self.compute_scores(output_transformer, mask_for_seq, enriched_all_item_embeddings)

        # 7. محاسبه SSL loss (اختیاری)
        ssl_loss_value = torch.tensor(0.0, device=scores.device)
        if is_train and self.ssl_weight > 0:
            try:
                # ssl_base_emb از seq_hidden_gnn_output (خروجی GNN محلی قبل از ترانسفورمر) گرفته می‌شود
                # یا از output_transformer (خروجی ترانسفورمر)
                # انتخاب seq_hidden_gnn_output منطقی‌تر به نظر می‌رسد برای SSL روی ساختار گراف
                last_idx_for_ssl = torch.clamp(mask_for_seq.sum(1) - 1, min=0).long()
                # اطمینان از اینکه ایندکس‌ها و تِنسور روی یک دستگاه هستند
                batch_indices_ssl = torch.arange(mask_for_seq.size(0), device=seq_hidden_gnn_output.device)
                last_idx_for_ssl = last_idx_for_ssl.to(seq_hidden_gnn_output.device)

                ssl_base_emb_seq = seq_hidden_gnn_output[batch_indices_ssl, last_idx_for_ssl]
                
                ssl_emb1 = F.dropout(ssl_base_emb_seq, p=self.ssl_dropout_rate, training=True)
                ssl_emb2 = F.dropout(ssl_base_emb_seq, p=self.ssl_dropout_rate, training=True)
                
                ssl_loss_value = self.calculate_ssl_loss(ssl_emb1, ssl_emb2, self.ssl_temp)
            except Exception as e:
                print(f"SSL calculation error: {e}")
                # ssl_loss_value از قبل صفر مقداردهی شده
        
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


# تابع wrapper forward که توسط train_test فراخوانی می‌شود
def forward(model: SessionGraph, i, data, is_train=True): # تایپ hint برای model
    # گرفتن داده‌های بچ از دیتا لودر
    alias_inputs_np, A_local_np, items_local_np, mask_seq_np, targets_np = data.get_slice(i)
    
    # تبدیل numpy arrayها به تنسور و انتقال به دستگاه مدل
    # این کار بهتر است یکبار انجام شود
    current_device = next(model.parameters()).device
    
    alias_inputs = torch.from_numpy(alias_inputs_np).long().to(current_device)
    A_local_adj = torch.from_numpy(A_local_np).float().to(current_device)
    items_local_ids = torch.from_numpy(items_local_np).long().to(current_device)
    mask_for_seq = torch.from_numpy(mask_seq_np).float().to(current_device)
    targets = torch.from_numpy(targets_np).long().to(current_device)
    
    # فراخوانی منطق اصلی مدل
    scores, ssl_loss = model.forward_model_logic(
        alias_inputs, 
        A_local_adj, 
        items_local_ids, 
        mask_for_seq, 
        is_train=is_train
    )
    
    return targets, scores, ssl_loss


def train_test(model, train_data, test_data, opt):
    # استفاده از autocast برای آموزش با دقت ترکیبی اگر CUDA در دسترس باشد
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp) 
    
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_ir = now_utc.astimezone(IR_TIMEZONE)
    print(f'Start training: {now_ir.strftime("%Y-%m-%d %H:%M:%S")}')
    
    model.train() # تنظیم مدل به حالت آموزش
    total_loss_epoch = 0.0
    total_rec_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0
    
    slices = train_data.generate_batch(opt.batchSize)
    num_batches = len(slices)

    for step, batch_indices in enumerate(slices):
        model.optimizer.zero_grad(set_to_none=True) # پاک کردن گرادیان‌ها
        
        # استفاده از autocast برای forward pass
        with autocast(enabled=use_amp):
            targets, scores, ssl_loss = forward(model, batch_indices, train_data, is_train=True)
            
            # محاسبه لاس توصیه (recommendation loss)
            # اطمینان از اینکه تارگت‌ها در محدوده هستند (بزرگتر از 0 و کوچکتر مساوی n_node)
            # و اینکه scores به درستی اندیس‌گذاری می‌شود (تارگت‌ها 1-based هستند، scores معمولا 0-based)
            valid_targets_mask = (targets > 0) & (targets < model.n_node) # تارگت 0 پدینگ است
            
            rec_loss = torch.tensor(0.0, device=scores.device)
            if valid_targets_mask.any():
                # تبدیل تارگت‌های 1-based به 0-based برای اندیس‌گذاری در scores
                # scores معمولا برای آیتم‌های 1 تا n_node-1 است (اگر پدینگ 0 باشد)
                # scores.size(1) باید n_node-1 باشد
                target_values_0_based = (targets[valid_targets_mask] - 1).clamp(0, scores.size(1) - 1)
                rec_loss = model.loss_function(scores[valid_targets_mask], target_values_0_based)
            
            # ترکیب لاس‌ها
            current_batch_loss = rec_loss + model.ssl_weight * ssl_loss
        
        # Backward pass با استفاده از GradScaler
        if use_amp:
            scaler.scale(current_batch_loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            current_batch_loss.backward()
            model.optimizer.step()
        
        # به‌روزرسانی آمار لاس‌ها
        total_loss_epoch += current_batch_loss.item()
        total_rec_loss_epoch += rec_loss.item()
        total_ssl_loss_epoch += ssl_loss.item() if isinstance(ssl_loss, torch.Tensor) else ssl_loss # ssl_loss ممکن است float باشد
        
        if (step + 1) % max(1, num_batches // 5) == 0 or step == num_batches - 1:
            avg_total_loss = total_loss_epoch / (step + 1)
            avg_rec_loss = total_rec_loss_epoch / (step + 1)
            avg_ssl_loss = total_ssl_loss_epoch / (step + 1)
            print(f'Epoch [{model.scheduler.last_epoch if hasattr(model,"scheduler") else 0}/{opt.epoch}] Batch [{step + 1}/{num_batches}] '
                  f'Total Loss: {avg_total_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}, SSL Loss: {avg_ssl_loss:.4f}')
    
    # به‌روزرسانی نرخ یادگیری
    if hasattr(model, 'scheduler'):
        model.scheduler.step()
    
    # --- بخش ارزیابی ---
    model.eval() # تنظیم مدل به حالت ارزیابی
    # مقادیر پیش‌فرض برای متریک‌ها
    # k_metric باید از opt گرفته شود یا مقدار پیش‌فرض داشته باشد
    k_metric = getattr(opt, 'k_metric', 20) # فرض k=20
    
    # اگر test_data خالی است، ارزیابی را انجام نده یا متریک‌های صفر برگردان
    if test_data is None or test_data.length == 0:
        print("No evaluation data provided. Skipping evaluation.")
        return 0.0, 0.0, 0.0 # Recall, MRR, Precision (همگی صفر)

    hit_at_k, mrr_at_k, precision_at_k = [], [], []
    
    test_slices = test_data.generate_batch(opt.batchSize)
    with torch.no_grad(): # غیرفعال کردن محاسبه گرادیان در زمان ارزیابی
        for batch_indices_test in test_slices:
            # forward pass برای داده‌های تست
            targets_test, scores_test, _ = forward(model, batch_indices_test, test_data, is_train=False)
            
            # گرفتن top-k پیش‌بینی‌ها
            # scores_test باید (B, num_candidates) باشد، که num_candidates = n_node-1
            _, top_k_indices_0_based = scores_test.topk(k_metric, dim=1) # (B, k)
            
            # تبدیل ایندکس‌های 0-based به IDهای آیتم 1-based
            top_k_item_ids = top_k_indices_0_based + 1 # (B, k)
            
            targets_test_np = targets_test.cpu().numpy() # (B,)
            top_k_item_ids_np = top_k_item_ids.cpu().numpy() # (B, k)

            for i in range(targets_test_np.shape[0]):
                target_item_id = targets_test_np[i]
                predicted_item_ids_at_k = top_k_item_ids_np[i]
                
                if target_item_id > 0 and target_item_id < model.n_node: # فقط تارگت‌های معتبر
                    # محاسبه Hit@k
                    if target_item_id in predicted_item_ids_at_k:
                        hit_at_k.append(1)
                        # محاسبه MRR@k
                        rank = np.where(predicted_item_ids_at_k == target_item_id)[0][0] + 1
                        mrr_at_k.append(1.0 / rank)
                    else:
                        hit_at_k.append(0)
                        mrr_at_k.append(0.0)
                    
                    # محاسبه Precision@k (تعداد آیتم‌های مرتبط پیش‌بینی شده تقسیم بر k)
                    # در اینجا چون فقط یک تارگت داریم، اگر hit رخ داده باشد، precision 1/k است.
                    # این ممکن است با تعریف استاندارد precision متفاوت باشد اگر بیش از یک آیتم مرتبط ممکن باشد.
                    # برای SR-GNN و موارد مشابه، معمولا Recall@k و MRR@k مهمترند.
                    # precision_at_k.append(hit_at_k[-1] / k_metric) # یک تعریف ساده

    # محاسبه میانگین متریک‌ها
    # اگر لیستی خالی باشد، np.mean خطا می‌دهد، پس بررسی می‌کنیم
    final_recall_at_k = np.mean(hit_at_k) * 100 if hit_at_k else 0.0
    final_mrr_at_k = np.mean(mrr_at_k) * 100 if mrr_at_k else 0.0
    # final_precision_at_k = np.mean(precision_at_k) * 100 if precision_at_k else 0.0
    
    print(f'Evaluation Results @{k_metric}: Recall: {final_recall_at_k:.4f}%, MRR: {final_mrr_at_k:.4f}%')
    
    return final_recall_at_k, final_mrr_at_k, 0.0 # برگرداندن صفر برای precision فعلا