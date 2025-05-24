import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
# TransformerEncoder و TransformerEncoderLayer دیگر مستقیماً استفاده نمی‌شوند، مگر اینکه بخواهیم آن‌ها را حفظ کنیم
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy # برای deepcopy در TargetAwareTransformerEncoder
from torch.cuda.amp import autocast, GradScaler

# -------------- 1. کلاس PositionalEncoding (بدون تغییر) --------------
class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (sequence length, batch size, embed dim) or (batch_size, sequence length, embed dim)
        # تطبیق با batch_first=True در Transformer
        if x.dim() == 3 and x.size(1) == self.pe.size(0) and x.size(2) == self.pe.size(2):
             # حالت (batch_size, sequence length, embed dim)
             x = x + self.pe[:x.size(1), :].squeeze(1) # self.pe shape (max_len, 1, d_model) -> (seq_len, d_model)
        elif x.dim() == 3 and x.size(0) == self.pe.size(0):
             # حالت (sequence length, batch size, embed dim)
             x = x + self.pe[:x.size(0), :]
        else:
            seq_len_dim_index = -1
            if x.dim() == 3:
                # اولویت با batch_first=True اگر ابعاد اجازه دهند
                if x.size(1) <= self.pe.size(0):
                    seq_len_dim_index = 1
                elif x.size(0) <= self.pe.size(0):
                    seq_len_dim_index = 0

            if seq_len_dim_index != -1:
                seq_len = x.size(seq_len_dim_index)
                if seq_len_dim_index == 1: # batch_first = True
                     # self.pe shape (max_len, 1, d_model) -> (seq_len, d_model) -> (1, seq_len, d_model)
                     pe_to_add = self.pe[:seq_len, :].squeeze(1).unsqueeze(0)
                else: # batch_first = False
                     pe_to_add = self.pe[:seq_len, :] # Shape: (seq_len, 1, d_model)

                try:
                    x = x + pe_to_add
                except RuntimeError as e:
                    # print(f"Warning: PositionalEncoding shape mismatch during addition. x: {x.shape}, pe_to_add: {pe_to_add.shape}. Error: {e}. Skipping addition.")
                    pass # ادامه بدون جمع، اگر ابعاد به طور دقیق تطابق نداشته باشند
            # else:
                # print(f"Warning: Could not determine sequence length or seq too long for PositionalEncoding. x shape: {x.shape}, max_len: {self.pe.size(0)}. Skipping addition.")
        return self.dropout(x)

# -------------- 2. کلاس GNN (بدون تغییر) --------------
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
        # self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True) # استفاده نشده

    def GNNCell(self, A, hidden):
        device = hidden.device
        A = A.to(device)

        if torch.isnan(A).any() or torch.isinf(A).any():
             # print("Warning: NaN/Inf detected in GNN input A.")
             A = torch.nan_to_num(A)
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
             # print("Warning: NaN/Inf detected in GNN input hidden.")
             hidden = torch.nan_to_num(hidden)

        A_in = A[:, :, :A.shape[1]]
        A_out = A[:, :, A.shape[1]: 2 * A.shape[1]]

        try:
            input_in = torch.matmul(A_in, self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A_out, self.linear_edge_out(hidden)) + self.b_oah
        except RuntimeError as e:
             print("Error during GNN matmul:")
             print("A_in shape:", A_in.shape); print("A_out shape:", A_out.shape)
             print("hidden shape:", hidden.shape)
             print("linear_edge_in(hidden) shape:", self.linear_edge_in(hidden).shape)
             raise e

        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        if torch.isnan(hy).any() or torch.isinf(hy).any():
             # print("Warning: NaN/Inf detected in GNN output hy.")
             hy = torch.nan_to_num(hy)
        return hy

    def forward(self, A, hidden):
        for _ in range(self.step): # _ به جای i برای نشان دادن عدم استفاده از مقدار
            hidden = self.GNNCell(A, hidden)
        return hidden

# -------------- 3. لایه انکودر آگاه از هدف (بدون تغییر) --------------
class TargetAwareEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super(TargetAwareEncoderLayer, self).__init__()
        # --- لایه‌های مربوط به کانتکست از کاندیداها ---
        self.W_sc_q = nn.Linear(d_model, d_model) # Query from src
        self.W_sc_k = nn.Linear(d_model, d_model) # Key from candidates
        self.W_sc_v = nn.Linear(d_model, d_model) # Value from candidates
        # self.context_projection = nn.Linear(d_model, d_model) # اگر ابعاد خروجی W_sc_v با d_model متفاوت باشد

        # --- لایه Self-Attention اصلی ---
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # --- لایه FeedForward ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, src, candidate_embeddings_global, src_mask=None, src_key_padding_mask=None):
        # src: (B, L, D) - توالی ورودی
        # candidate_embeddings_global: (N, D) - امبدینگ تمام کاندیداها
        # src_key_padding_mask: (B, L) - ماسک برای پدینگ در src

        # 1. محاسبه کانتکست از کاندیداها برای هر آیتم توالی
        q_sc = self.W_sc_q(src)                            # (B, L, D)
        k_sc = self.W_sc_k(candidate_embeddings_global)    # (N, D)
        v_sc = self.W_sc_v(candidate_embeddings_global)    # (N, D)

        # attn_weights_sc: (B, L, N) - وزن توجه هر آیتم توالی به هر کاندیدا
        attn_score_sc = torch.matmul(q_sc, k_sc.transpose(0, 1)) / math.sqrt(q_sc.size(-1))
        attn_weights_sc = F.softmax(attn_score_sc, dim=-1)

        # context_from_candidates: (B, L, D) - کانتکست وزندار شده از کاندیداها
        context_from_candidates = torch.matmul(attn_weights_sc, v_sc)
        # projected_context = self.context_projection(context_from_candidates) # در صورت نیاز به تغییر ابعاد

        # 2. افزایش نمایش توالی با کانتکست به‌دست‌آمده
        src_enhanced = src + context_from_candidates # ترکیب ساده با جمع

        # 3. Self-Attention اصلی روی src_enhanced
        # nn.MultiheadAttention انتظار Q, K, V را دارد. اینجا از src_enhanced مشتق می‌شوند.
        sa_output, _ = self.self_attn(src_enhanced, src_enhanced, src_enhanced,
                                      key_padding_mask=src_key_padding_mask,
                                      attn_mask=src_mask) # sa_output is (B, L, D)

        # اولین اتصال باقیمانده و نرمال‌سازی (با src اصلی)
        out1 = src + self.dropout1(sa_output)
        out1 = self.norm1(out1)

        # 4. لایه FeedForward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(out1))))

        # دومین اتصال باقیمانده و نرمال‌سازی
        out2 = out1 + self.dropout2(ff_output)
        out2 = self.norm2(out2)

        return out2

# -------------- 4. انکودر ترانسفورمر آگاه از هدف (بدون تغییر) --------------
class TargetAwareTransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TargetAwareTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, candidate_embeddings_global, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, candidate_embeddings_global, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

# -------------- 5. کلاس SessionGraph (تغییر یافته برای انکودر جدید - بدون تغییر در این مرحله) --------------
class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, step=opt.step)

        self.ssl_weight = opt.ssl_weight
        self.ssl_temp = opt.ssl_temp
        self.ssl_dropout_rate = opt.ssl_dropout_rate

        # --- لایه‌های انکودر ترانسفورمر آگاه از هدف ---
        self.pos_encoder = PositionalEncoding(self.hidden_size, getattr(opt, 'dropout', 0.1), max_len=5000)

        ta_encoder_layer = TargetAwareEncoderLayer(
            d_model=self.hidden_size,
            nhead=getattr(opt, 'nhead', 2),
            dim_feedforward=getattr(opt, 'ff_hidden', 256),
            dropout=getattr(opt, 'dropout', 0.1)
        )
        self.transformer_encoder = TargetAwareTransformerEncoder( # نام متغیر می‌تواند همین بماند
            encoder_layer=ta_encoder_layer,
            num_layers=getattr(opt, 'nlayers', 2),
            norm=nn.LayerNorm(self.hidden_size) # نرمال‌سازی نهایی اختیاری
        )
        # -----------------------------------------

        # --- لایه های محاسبه امتیاز (بدون تغییر عمده نسبت به قبل، روی خروجی انکودر جدید عمل می‌کنند) ---
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # برای بخش Target Attention در compute_scores
        # ---------------------------

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.uniform_(weight, -stdv, stdv)
        nn.init.uniform_(self.embedding.weight, -stdv, stdv)
        if self.embedding.padding_idx is not None:
             with torch.no_grad():
                  self.embedding.weight[self.embedding.padding_idx].fill_(0)


    def compute_scores(self, hidden_transformer_output, mask):
        # hidden_transformer_output از TargetAwareTransformerEncoder می‌آید
        # mask مربوط به توالی اصلی است
        device = hidden_transformer_output.device
        batch_size = hidden_transformer_output.size(0)

        # --- محاسبه ht (آخرین آیتم) و a (ترجیح جهانی جلسه) ---
        sequence_lengths = torch.sum(mask.float(), 1).long()
        ht = torch.zeros(batch_size, self.hidden_size, device=device)
        valid_lengths_mask = sequence_lengths > 0

        if valid_lengths_mask.any(): # فقط اگر طول‌های معتبر وجود داشته باشند
            # اطمینان از اینکه gather_indices منفی نشود
            gather_indices = (sequence_lengths[valid_lengths_mask] - 1).clamp(min=0)
            batch_indices_ht = torch.arange(batch_size, device=device)[valid_lengths_mask]
            # اطمینان از اینکه hidden_transformer_output حداقل یک آیتم در بعد توالی دارد اگر valid_lengths_mask.any() برقرار است
            if hidden_transformer_output.size(1) > 0 :
                 ht[valid_lengths_mask] = hidden_transformer_output[batch_indices_ht, gather_indices]
            # else:
                 # print(f"Warning: hidden_transformer_output is empty in seq dim ({hidden_transformer_output.shape}) but expected valid lengths. ht remains zero.")


        q1 = self.linear_one(ht).view(batch_size, 1, self.hidden_size)
        q2 = self.linear_two(hidden_transformer_output)

        mask_expanded_alpha = mask.unsqueeze(-1).float()
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2))
        # اطمینان از اینکه hidden_transformer_output برای ماسک کردن خالی نیست
        if hidden_transformer_output.size(1) == 0 and mask_expanded_alpha.size(1) > 0 :
             # print(f"Warning: Mismatch in sequence length for alpha masking. alpha_logits: {alpha_logits.shape}, mask: {mask_expanded_alpha.shape}")
             # اگر hidden_transformer_output خالی است، alpha_logits هم باید خالی باشد یا ماسک مناسب باشد.
             # این حالت نباید رخ دهد اگر داده‌ها و پردازش‌ها درست باشند.
             alpha_logits_masked = alpha_logits # رد شدن از ماسک اگر ابعاد ناجور باشند
        else:
            alpha_logits_masked = alpha_logits.masked_fill(mask_expanded_alpha == 0, -float('inf'))

        alpha = F.softmax(alpha_logits_masked, dim=1)

        # اطمینان از اینکه alpha و hidden_transformer_output دارای بعد توالی یکسان هستند
        if alpha.size(1) == hidden_transformer_output.size(1) and hidden_transformer_output.size(1) > 0:
             a = torch.sum(alpha * hidden_transformer_output * mask_expanded_alpha, 1)
        elif hidden_transformer_output.size(1) == 0 : # اگر توالی خالی باشد، a هم صفر است
             a = torch.zeros(batch_size, self.hidden_size, device=device)
        else: # اگر طول‌ها متفاوت باشند (نباید اتفاق بیفتد)
            # print(f"Warning: Sequence length mismatch for 'a' calculation. alpha: {alpha.shape}, hidden: {hidden_transformer_output.shape}. Using zero for 'a'.")
            a = torch.zeros(batch_size, self.hidden_size, device=device)


        candidate_embeds = self.embedding.weight[1:].to(device) # (n_node-1, hidden_size)

        if self.nonhybrid:
            combined_preference = self.linear_transform(torch.cat([a, ht], 1))
            scores = torch.matmul(combined_preference, candidate_embeds.t())
        else:
            # --- بخش Target Attention (مانند قبل، روی خروجی انکودر جدید عمل می‌کند) ---
            mask_expanded_beta = mask.unsqueeze(-1).float() # (b, s, 1)
            # اطمینان از اینکه hidden_transformer_output خالی نیست قبل از ارسال به linear_t
            if hidden_transformer_output.size(1) > 0:
                 hidden_masked_for_qt = hidden_transformer_output * mask_expanded_beta
                 qt = self.linear_t(hidden_masked_for_qt) # (b, s, d)
            else: # اگر توالی خالی باشد
                 qt = torch.zeros_like(hidden_transformer_output) # (b, 0, d)

            # beta_logits: (b, n_node-1, s)
            # اگر qt خالی باشد (بعد توالی صفر)، transpose(1,2) خطا میدهد اگر s=0.
            if qt.size(1) > 0: # s > 0
                 beta_logits = torch.matmul(candidate_embeds, qt.transpose(1, 2))
                 # ماسک برای سافت‌مکس بتا: (b, 1, s)
                 beta_mask = mask.unsqueeze(1) # (b, 1, s)
                 beta_logits_masked = beta_logits.masked_fill(beta_mask == 0, -float('inf'))
                 beta = F.softmax(beta_logits_masked, dim=-1) # (b, n_node-1, s)

                 # target_ctx: (b, n_node-1, d)
                 # qt * mask_expanded_beta باید انجام شود اگر qt خالی نیست
                 target_ctx = torch.matmul(beta, qt * mask_expanded_beta) # qt از قبل ماسک شده است، اما برای اطمینان
            else: # اگر qt خالی باشد (بعد توالی صفر)
                 target_ctx = torch.zeros(batch_size, candidate_embeds.size(0), self.hidden_size, device=device)


            final_representation = a.unsqueeze(1) + target_ctx
            scores = torch.sum(final_representation * candidate_embeds.unsqueeze(0), dim=-1)

        return scores

    def calculate_ssl_loss(self, emb1, emb2, temperature):
        # ... (بدون تغییر) ...
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        sim_matrix_12 = torch.matmul(emb1, emb2.t()) / temperature
        log_softmax_12 = F.log_softmax(sim_matrix_12, dim=1)
        loss_12 = -torch.diag(log_softmax_12)
        sim_matrix_21 = torch.matmul(emb2, emb1.t()) / temperature
        log_softmax_21 = F.log_softmax(sim_matrix_21, dim=1)
        loss_21 = -torch.diag(log_softmax_21)
        ssl_loss = (loss_12.mean() + loss_21.mean()) / 2.0
        return ssl_loss

# (توابع trans_to_cuda, trans_to_cpu بدون تغییر)
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

# -------------- تابع forward کلی (بدون تغییر در این مرحله) --------------
def forward(model, i, data, is_train=True):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    # ... (انتقال به CUDA مانند قبل) ...
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items_gnn_input = trans_to_cuda(torch.Tensor(items).long()) # تغییر نام items به items_gnn_input برای وضوح
    A_gnn = trans_to_cuda(torch.tensor(np.array(A), dtype=torch.float32)) # تغییر نام A به A_gnn
    mask_seq = trans_to_cuda(torch.Tensor(mask).long()) # تغییر نام mask به mask_seq
    targets_final = trans_to_cuda(torch.Tensor(targets).long()) # تغییر نام targets به targets_final

    # 1. Embedding Layer برای GNN
    hidden_emb = model.embedding(items_gnn_input) # (batch, max_n_node_in_batch, hidden_size)
    if torch.isnan(hidden_emb).any() or torch.isinf(hidden_emb).any():
        # print(f"Warning: NaN/Inf detected after embedding layer for batch slice {i}.")
        hidden_emb = torch.nan_to_num(hidden_emb)

    # 2. GNN Layer
    hidden_gnn = model.gnn(A_gnn, hidden_emb) # (batch, max_n_node_in_batch, hidden_size)
    if torch.isnan(hidden_gnn).any() or torch.isinf(hidden_gnn).any():
        # print(f"Warning: NaN/Inf detected after GNN layer for batch slice {i}.")
        hidden_gnn = torch.nan_to_num(hidden_gnn)

    batch_size_fwd, _ = alias_inputs.shape # _ به جای seq_len_max_in_data_get_slice

    # 3. استخراج بازنمایی‌های توالی برای انکودر ترانسفورمر
    clamped_alias_inputs = alias_inputs.clamp(0, hidden_gnn.size(1) - 1)
    alias_expanded_for_gather = clamped_alias_inputs.unsqueeze(-1).expand(-1, -1, model.hidden_size)
    seq_hidden_gnn = torch.gather(hidden_gnn, 1, alias_expanded_for_gather) # (B, SeqLenOriginalPadded, D)

    # --- مسیر اصلی توصیه ---
    # 4. Positional Encoding
    seq_hidden_pos = model.pos_encoder(seq_hidden_gnn)

    # 5. Target-Aware Transformer Encoder
    src_key_padding_mask = (mask_seq == 0) # (batch_size, seq_length)

    # دریافت امبدینگ کاندیداها برای انکودر
    candidate_embeds_global = model.embedding.weight[1:].to(seq_hidden_pos.device) # (N-1, D)

    hidden_transformer_output = model.transformer_encoder(
        src=seq_hidden_pos,
        candidate_embeddings_global=candidate_embeds_global, # پاس دادن کاندیداها
        src_key_padding_mask=src_key_padding_mask
    ) # Output: (batch, seq, feature)

    # 6. محاسبه امتیازات توصیه
    scores = model.compute_scores(hidden_transformer_output, mask_seq)

    # --- مسیر SSL (فقط در حالت آموزش) ---
    ssl_loss = torch.tensor(0.0, device=scores.device)
    if is_train and model.ssl_weight > 0:
        try:
            sequence_lengths_ssl = torch.sum(mask_seq.float(), 1).long()
            batch_indices_ssl = torch.arange(batch_size_fwd, device=alias_inputs.device) # استفاده از batch_size_fwd
            valid_lengths_mask_ssl = sequence_lengths_ssl > 0

            last_item_node_indices_in_gnn = torch.zeros(batch_size_fwd, dtype=torch.long, device=alias_inputs.device)

            if valid_lengths_mask_ssl.any():
                 last_valid_seq_indices = (sequence_lengths_ssl[valid_lengths_mask_ssl] - 1).clamp(min=0)
                 last_item_node_indices_in_gnn[valid_lengths_mask_ssl] = alias_inputs[batch_indices_ssl[valid_lengths_mask_ssl], last_valid_seq_indices]

            clamped_last_item_node_indices = last_item_node_indices_in_gnn.clamp(0, hidden_gnn.size(1) - 1)

            ssl_base_emb = torch.zeros(batch_size_fwd, model.hidden_size, device=hidden_gnn.device)
            if valid_lengths_mask_ssl.any() and hidden_gnn.size(1) > 0: # اطمینان از اینکه hidden_gnn خالی نیست
                ssl_base_emb[valid_lengths_mask_ssl] = hidden_gnn[batch_indices_ssl[valid_lengths_mask_ssl], clamped_last_item_node_indices[valid_lengths_mask_ssl]]

            ssl_emb1 = F.dropout(ssl_base_emb, p=model.ssl_dropout_rate, training=True)
            ssl_emb2 = F.dropout(ssl_base_emb, p=model.ssl_dropout_rate, training=True)
            ssl_loss = model.calculate_ssl_loss(ssl_emb1, ssl_emb2, model.ssl_temp)

            if torch.isnan(ssl_loss).any() or torch.isinf(ssl_loss).any():
                 # print(f"Warning: NaN/Inf detected in SSL loss for batch slice {i}. Setting to 0.")
                 ssl_loss = torch.tensor(0.0, device=scores.device)
        except Exception as e:
             print(f"Error during SSL calculation for batch slice {i}: {e}")
             # import traceback; traceback.print_exc() # برای دیباگ بیشتر
             ssl_loss = torch.tensor(0.0, device=scores.device)

    return targets_final.to(scores.device), scores, ssl_loss


# -------------- تابع آموزش و تست (تغییر یافته برای Precision@20) --------------
def train_test(model, train_data, test_data, opt): # اضافه کردن opt
    scaler = GradScaler(enabled=torch.cuda.is_available())

    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    total_rec_loss = 0.0
    total_ssl_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for step, i_slice in enumerate(slices): # تغییر نام i به i_slice
        model.optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            targets, scores, ssl_loss_val = forward(model, i_slice, train_data, is_train=True) # تغییر نام ssl_loss به ssl_loss_val

            valid_targets_mask = (targets > 0) & (targets <= model.n_node)
            rec_loss = torch.tensor(0.0, device=scores.device) # مقدار پیش‌فرض

            if valid_targets_mask.sum() > 0:
                 if scores.shape[1] == model.n_node - 1:
                      try:
                           target_values_for_loss = (targets[valid_targets_mask] - 1).clamp(0, scores.shape[1] - 1)
                           rec_loss = model.loss_function(scores[valid_targets_mask], target_values_for_loss)
                      except IndexError as e:
                            # print(f"Error in CrossEntropyLoss training slice {step}: {e}")
                            # print(f"Scores shape: {scores.shape}, Max target-1: {(targets[valid_targets_mask]-1).max()}")
                            pass # rec_loss صفر باقی می‌ماند
                 else:
                      # print(f"Warning: Score dim mismatch in training. scores: {scores.shape[1]}, expected: {model.n_node - 1}")
                      pass
            else:
                # print(f"Warning: No valid targets in training batch slice {step}.")
                pass


            loss = rec_loss + model.ssl_weight * ssl_loss_val

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                 # print(f"Warning: NaN/Inf detected in total loss for training slice {step}. Skipping batch.")
                 continue

        if torch.cuda.is_available():
             scaler.scale(loss).backward()
             scaler.step(model.optimizer)
             scaler.update()
        else:
             loss.backward()
             model.optimizer.step()

        total_loss += loss.item() if not torch.isnan(loss).any() else 0
        total_rec_loss += rec_loss.item() if not torch.isnan(rec_loss).any() else 0
        total_ssl_loss += ssl_loss_val.item() if not torch.isnan(ssl_loss_val).any() else 0


        if (step + 1) % max(1, int(len(slices) / 5)) == 0: # لاگ حدود 5 بار
             avg_loss = total_loss / (step + 1) if (step+1) > 0 else 0
             avg_rec_loss = total_rec_loss / (step + 1) if (step+1) > 0 else 0
             avg_ssl_loss = total_ssl_loss / (step + 1) if (step+1) > 0 else 0
             print('[%d/%d] Tot Loss: %.4f, Rec Loss: %.4f, SSL Loss: %.4f' %
                   (step + 1, len(slices), avg_loss, avg_rec_loss, avg_ssl_loss))

    model.scheduler.step()

    len_slices = len(slices) if slices else 1 # جلوگیری از تقسیم بر صفر
    print('\tAvg Loss:\t%.4f' % (total_loss / len_slices))
    print('\tAvg Rec Loss:\t%.4f' % (total_rec_loss / len_slices))
    print('\tAvg SSL Loss:\t%.4f' % (total_ssl_loss / len_slices))


    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr, precision = [], [], [] # اضافه شدن لیست precision
    k_metric = 20 # مقدار k برای Recall@k, MRR@k, Precision@k

    test_slices_eval = test_data.generate_batch(model.batch_size) # تغییر نام slices
    with torch.no_grad():
        for k_slice_eval in test_slices_eval: # تغییر نام i
            targets, scores, _ = forward(model, k_slice_eval, test_data, is_train=False)
            sub_scores_indices = scores.topk(k_metric)[1]
            sub_scores_indices_cpu = trans_to_cpu(sub_scores_indices).detach().numpy()
            targets_cpu = trans_to_cpu(targets).detach().numpy()

            for score_idx_list, target_item_id in zip(sub_scores_indices_cpu, targets_cpu):
                 if target_item_id > 0:
                    target_item_id_zero_based = target_item_id - 1
                    is_hit = np.isin(target_item_id_zero_based, score_idx_list)
                    hit.append(is_hit)

                    if is_hit:
                        rank_list = np.where(score_idx_list == target_item_id_zero_based)[0]
                        rank = rank_list[0] + 1 if len(rank_list) > 0 else float('inf')
                        mrr.append(1.0 / rank)
                        precision.append(1.0 / k_metric) # اگر آیتم هدف پیدا شد
                    else:
                        mrr.append(0.0)
                        precision.append(0.0) # اگر آیتم هدف پیدا نشد

    hit_rate = np.mean(hit) * 100 if hit else 0.0
    mrr_score = np.mean(mrr) * 100 if mrr else 0.0
    precision_score = np.mean(precision) * 100 if precision else 0.0 # محاسبه میانگین precision

    return hit_rate, mrr_score, precision_score # برگرداندن precision_score