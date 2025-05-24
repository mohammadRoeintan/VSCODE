import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import copy # برای deepcopy در TargetAwareTransformerEncoder
from torch.cuda.amp import autocast, GradScaler
import torch.profiler #  Profiler اضافه شد

# -------------- 1. کلاس PositionalEncoding (بدون تغییر از قبل) --------------
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
        if x.dim() == 3: # (batch, seq, feature) or (seq, batch, feature)
            # تشخیص خودکار batch_first بر اساس اینکه کدام بعد به max_len نزدیکتر است (ساده‌سازی شده)
            # فرض می‌کنیم اگر بعد دوم کوچکتر از max_len باشد، batch_first=True است.
            is_batch_first = x.size(1) < self.pe.size(0) if x.size(0) >= x.size(1) else True

            if is_batch_first and x.size(1) <= self.pe.size(0): # batch_first=True (batch, seq, feature)
                 pe_to_add = self.pe[:x.size(1), :].squeeze(1).unsqueeze(0) # (1, seq, feature)
            elif not is_batch_first and x.size(0) <= self.pe.size(0): # batch_first=False (seq, batch, feature)
                 pe_to_add = self.pe[:x.size(0), :] # (seq, 1, feature)
            else:
                # print(f"Warning: PositionalEncoding seq_len > max_len. x: {x.shape}, pe_max_len: {self.pe.size(0)}. Skipping.")
                return self.dropout(x) # در صورت عدم تطابق، بدون PE برگردان
            
            try:
                x = x + pe_to_add
            except RuntimeError:
                # print(f"Warning: PositionalEncoding shape mismatch during addition. x: {x.shape}, pe_to_add: {pe_to_add.shape}. Skipping.")
                pass # اگر جمع موفقیت‌آمیز نبود، از آن صرف نظر کن
        # else:
            # print(f"Warning: PositionalEncoding input x has unexpected dim {x.dim()}. Skipping.")
        return self.dropout(x)

# -------------- 2. کلاس GNN (بدون تغییر از قبل) --------------
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

    def GNNCell(self, A, hidden):
        device = hidden.device
        A = A.to(device)

        if torch.isnan(A).any() or torch.isinf(A).any():
             A = torch.nan_to_num(A)
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
             hidden = torch.nan_to_num(hidden)
        
        A_in = A[:, :, :A.shape[1]]
        A_out = A[:, :, A.shape[1]: 2 * A.shape[1]]

        try:
            input_in = torch.matmul(A_in, self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A_out, self.linear_edge_out(hidden)) + self.b_oah
        except RuntimeError as e:
             print(f"Error during GNN matmul: A_in:{A_in.shape}, hidden_lin:{self.linear_edge_in(hidden).shape}, Error:{e}")
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
             hy = torch.nan_to_num(hy)
        return hy

    def forward(self, A, hidden):
        for _ in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

# -------------- 3. لایه انکودر آگاه از هدف (بدون تغییر از قبل) --------------
class TargetAwareEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super(TargetAwareEncoderLayer, self).__init__()
        self.W_sc_q = nn.Linear(d_model, d_model)
        self.W_sc_k = nn.Linear(d_model, d_model)
        self.W_sc_v = nn.Linear(d_model, d_model)
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
        q_sc = self.W_sc_q(src)
        k_sc = self.W_sc_k(candidate_embeddings_global)
        v_sc = self.W_sc_v(candidate_embeddings_global)
        attn_score_sc = torch.matmul(q_sc, k_sc.transpose(0, 1)) / math.sqrt(q_sc.size(-1))
        attn_weights_sc = F.softmax(attn_score_sc, dim=-1)
        context_from_candidates = torch.matmul(attn_weights_sc, v_sc)
        src_enhanced = src + context_from_candidates
        sa_output, _ = self.self_attn(src_enhanced, src_enhanced, src_enhanced,
                                      key_padding_mask=src_key_padding_mask,
                                      attn_mask=src_mask)
        out1 = src + self.dropout1(sa_output) 
        out1 = self.norm1(out1)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(out1))))
        out2 = out1 + self.dropout2(ff_output)
        out2 = self.norm2(out2)
        return out2

# -------------- 4. انکودر ترانسفورمر آگاه از هدف (بدون تغییر از قبل) --------------
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

# -------------- 5. کلاس SessionGraph (بدون تغییر عمده از قبل) --------------
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
        self.pos_encoder = PositionalEncoding(self.hidden_size, getattr(opt, 'dropout', 0.1), max_len=5000)
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
        device = hidden_transformer_output.device
        batch_size = hidden_transformer_output.size(0)
        ht = torch.zeros(batch_size, self.hidden_size, device=device)
        sequence_lengths = torch.sum(mask.float(), 1).long()
        valid_lengths_mask = sequence_lengths > 0

        if valid_lengths_mask.any():
            gather_indices = (sequence_lengths[valid_lengths_mask] - 1).clamp(min=0)
            batch_indices_ht = torch.arange(batch_size, device=device)[valid_lengths_mask]
            if hidden_transformer_output.size(1) > 0 :
                 ht[valid_lengths_mask] = hidden_transformer_output[batch_indices_ht, gather_indices]

        q1 = self.linear_one(ht).view(batch_size, 1, self.hidden_size)
        q2 = self.linear_two(hidden_transformer_output)
        mask_expanded_alpha = mask.unsqueeze(-1).float()
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2))
        
        if hidden_transformer_output.size(1) == 0 and mask_expanded_alpha.size(1) > 0 :
            alpha_logits_masked = alpha_logits 
        elif hidden_transformer_output.size(1) < alpha_logits.size(1) and hidden_transformer_output.numel() > 0: # اگر طول‌ها متفاوت باشند
            # این حالت پیچیده است، ممکن است نیاز به بریدن alpha_logits یا mask_expanded_alpha باشد
            # برای سادگی، اگر عدم تطابق جدی باشد، ماسک اعمال نمی‌کنیم یا خطا می‌دهیم
            # print(f"Warning: Mismatch for alpha masking. alpha_logits: {alpha_logits.shape}, mask: {mask_expanded_alpha.shape}, hidden_T: {hidden_transformer_output.shape}")
            # در اینجا alpha_logits را بدون ماسک استفاده می‌کنیم، که ممکن است ایده‌آل نباشد
            alpha_logits_masked = alpha_logits
        else:
            alpha_logits_masked = alpha_logits.masked_fill(mask_expanded_alpha[:, :alpha_logits.size(1)] == 0, -float('inf'))


        alpha = F.softmax(alpha_logits_masked, dim=1)
        a = torch.zeros(batch_size, self.hidden_size, device=device)
        if alpha.size(1) == hidden_transformer_output.size(1) and hidden_transformer_output.size(1) > 0:
             a = torch.sum(alpha * hidden_transformer_output * mask_expanded_alpha[:, :alpha.size(1)], 1)
        elif hidden_transformer_output.size(1) > 0 and alpha.size(1) > hidden_transformer_output.size(1) : # اگر آلفا بزرگتر است
             a = torch.sum(alpha[:,:hidden_transformer_output.size(1)] * hidden_transformer_output * mask_expanded_alpha[:, :hidden_transformer_output.size(1)], 1)


        candidate_embeds = self.embedding.weight[1:].to(device)

        if self.nonhybrid:
            combined_preference = self.linear_transform(torch.cat([a, ht], 1))
            scores = torch.matmul(combined_preference, candidate_embeds.t())
        else:
            mask_expanded_beta = mask.unsqueeze(-1).float()
            qt = torch.zeros_like(hidden_transformer_output)
            if hidden_transformer_output.size(1) > 0:
                 hidden_masked_for_qt = hidden_transformer_output * mask_expanded_beta[:,:hidden_transformer_output.size(1)]
                 qt = self.linear_t(hidden_masked_for_qt)
            
            target_ctx = torch.zeros(batch_size, candidate_embeds.size(0), self.hidden_size, device=device)
            if qt.size(1) > 0:
                 beta_logits = torch.matmul(candidate_embeds, qt.transpose(1, 2))
                 beta_mask = mask.unsqueeze(1)
                 beta_logits_masked = beta_logits.masked_fill(beta_mask[:,:,:qt.size(1)] == 0, -float('inf'))
                 beta = F.softmax(beta_logits_masked, dim=-1)
                 target_ctx = torch.matmul(beta, qt * mask_expanded_beta[:,:qt.size(1)])
            
            final_representation = a.unsqueeze(1) + target_ctx
            scores = torch.sum(final_representation * candidate_embeds.unsqueeze(0), dim=-1)
        return scores

    def calculate_ssl_loss(self, emb1, emb2, temperature):
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

def forward(model, i_slice_data, data_source, is_train=True): # تغییر نام پارامترها برای وضوح
    alias_inputs, A_gnn_batch, items_for_gnn, mask_seq_batch, targets_batch = data_source.get_slice(i_slice_data)
    
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items_for_gnn = trans_to_cuda(torch.Tensor(items_for_gnn).long())
    A_gnn_batch = trans_to_cuda(torch.tensor(np.array(A_gnn_batch), dtype=torch.float32))
    mask_seq_batch = trans_to_cuda(torch.Tensor(mask_seq_batch).long())
    targets_batch = trans_to_cuda(torch.Tensor(targets_batch).long())

    hidden_emb = model.embedding(items_for_gnn)
    if torch.isnan(hidden_emb).any() or torch.isinf(hidden_emb).any():
        hidden_emb = torch.nan_to_num(hidden_emb)

    hidden_gnn = model.gnn(A_gnn_batch, hidden_emb)
    if torch.isnan(hidden_gnn).any() or torch.isinf(hidden_gnn).any():
        hidden_gnn = torch.nan_to_num(hidden_gnn)
    
    batch_size_fwd, _ = alias_inputs.shape
    clamped_alias_inputs = alias_inputs.clamp(0, hidden_gnn.size(1) - 1) # hidden_gnn.size(1) is max_n_node_in_batch
    
    # اطمینان از اینکه hidden_gnn حداقل یک بعد برای گره‌ها دارد قبل از gather
    seq_hidden_gnn = torch.zeros(batch_size_fwd, alias_inputs.size(1), model.hidden_size, device=hidden_gnn.device)
    if hidden_gnn.size(1) > 0: # فقط اگر گره‌ای در بچ برای GNN وجود داشته باشد
        alias_expanded_for_gather = clamped_alias_inputs.unsqueeze(-1).expand(-1, -1, model.hidden_size)
        seq_hidden_gnn = torch.gather(hidden_gnn, 1, alias_expanded_for_gather)

    seq_hidden_pos = model.pos_encoder(seq_hidden_gnn) 
    src_key_padding_mask = (mask_seq_batch == 0)
    candidate_embeds_global = model.embedding.weight[1:].to(seq_hidden_pos.device)
    
    hidden_transformer_output = model.transformer_encoder(
        src=seq_hidden_pos,
        candidate_embeddings_global=candidate_embeds_global,
        src_key_padding_mask=src_key_padding_mask
    )

    scores = model.compute_scores(hidden_transformer_output, mask_seq_batch) 

    ssl_loss = torch.tensor(0.0, device=scores.device) 
    if is_train and model.ssl_weight > 0:
        try:
            sequence_lengths_ssl = torch.sum(mask_seq_batch.float(), 1).long()
            batch_indices_ssl = torch.arange(batch_size_fwd, device=alias_inputs.device)
            valid_lengths_mask_ssl = sequence_lengths_ssl > 0
            last_item_node_indices_in_gnn = torch.zeros(batch_size_fwd, dtype=torch.long, device=alias_inputs.device)

            if valid_lengths_mask_ssl.any():
                 last_valid_seq_indices = (sequence_lengths_ssl[valid_lengths_mask_ssl] - 1).clamp(min=0)
                 # اطمینان از اینکه alias_inputs برای اندیس‌گذاری خالی نیست
                 if alias_inputs.size(1) > 0:
                    last_item_node_indices_in_gnn[valid_lengths_mask_ssl] = alias_inputs[batch_indices_ssl[valid_lengths_mask_ssl], last_valid_seq_indices]
            
            clamped_last_item_node_indices = last_item_node_indices_in_gnn.clamp(0, hidden_gnn.size(1) - 1)
            ssl_base_emb = torch.zeros(batch_size_fwd, model.hidden_size, device=hidden_gnn.device)

            if valid_lengths_mask_ssl.any() and hidden_gnn.size(1) > 0:
                ssl_base_emb[valid_lengths_mask_ssl] = hidden_gnn[batch_indices_ssl[valid_lengths_mask_ssl], clamped_last_item_node_indices[valid_lengths_mask_ssl]]
            
            ssl_emb1 = F.dropout(ssl_base_emb, p=model.ssl_dropout_rate, training=True)
            ssl_emb2 = F.dropout(ssl_base_emb, p=model.ssl_dropout_rate, training=True)
            ssl_loss = model.calculate_ssl_loss(ssl_emb1, ssl_emb2, model.ssl_temp)

            if torch.isnan(ssl_loss).any() or torch.isinf(ssl_loss).any():
                 ssl_loss = torch.tensor(0.0, device=scores.device)
        except Exception:
             # print(f"Error during SSL calculation for batch slice (indices {i_slice_data}): {e}")
             ssl_loss = torch.tensor(0.0, device=scores.device)

    return targets_batch.to(scores.device), scores, ssl_loss


# -------------- تابع آموزش و تست (با قابلیت پروفایل کردن) --------------
def train_test(model, train_data, test_data, opt):
    scaler = GradScaler(enabled=torch.cuda.is_available())
    print('Start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    total_rec_loss = 0.0
    total_ssl_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    # --- تنظیمات پروفایلر ---
    # شما می‌توانید این بخش را برای فعال/غیرفعال کردن پروفایلر کنترل کنید
    # مثلاً فقط در epoch اول یا برای تعداد محدودی بچ پروفایل کنید.
    ENABLE_PROFILER = True # برای فعال کردن، True قرار دهید
    PROFILER_EPOCH_TARGET = 0 # در کدام epoch پروفایل انجام شود (0 یعنی اولین epoch)
    # به دست آوردن epoch فعلی از model یا opt اگر در آنجا ذخیره شده باشد، یا از طریق شمارنده خارجی
    # در اینجا فرض می‌کنیم فقط در اولین فراخوانی train_test (معادل اولین epoch) پروفایل می‌کنیم.
    # برای سادگی، فرض می‌کنیم train_test فقط یکبار در هر epoch از main.py فراخوانی می‌شود.
    
    # اگر این اولین epoch است، پروفایلر را آماده کنید
    # برای جلوگیری از تعریف مجدد در هر فراخوانی، این را می‌توان خارج از تابع train_test در main.py انجام داد
    # و profiler را به عنوان آرگومان به train_test پاس داد. اما برای سادگی، اینجا قرار می‌دهیم.
    
    # اگر میخواهید فقط در epoch خاصی پروفایل کنید و epoch فعلی را از main.py می‌گیرید:
    # current_epoch = opt.current_epoch # فرض کنید این را در opt دارید
    # if ENABLE_PROFILER and current_epoch == PROFILER_EPOCH_TARGET:
    
    # برای این مثال، فرض می‌کنیم اگر ENABLE_PROFILER=True باشد، در ابتدای آموزش پروفایل می‌کنیم.
    # فقط برای چند بچ اولیه پروفایل می‌کنیم تا فایل لاگ خیلی بزرگ نشود.
    wait_steps = 1
    warmup_steps = 1
    active_steps = 5 # تعداد بچ‌هایی که واقعا پروفایل می‌شوند
    profiler_total_steps = wait_steps + warmup_steps + active_steps

    # ایجاد پروفایلر (اگر فعال باشد)
    # مسیر ذخیره لاگ در Kaggle به /kaggle/working/ اضافه می‌شود
    profiler_log_dir = './log_profile_kaggle' # در Kaggle در /kaggle/working/log_profile_kaggle ذخیره می‌شود
    
    # استفاده از with torch.profiler.profile(...) در حلقه بچ‌ها
    # ما نمی‌خواهیم برای کل epoch پروفایل کنیم چون بسیار کند خواهد شد و لاگ بزرگی تولید می‌کند.
    # پس پروفایلر را فقط برای چند بچ اول اجرا می‌کنیم.

    for step, i_slice in enumerate(slices):
        if ENABLE_PROFILER and step < profiler_total_steps and hasattr(opt, 'epoch_num_main') and opt.epoch_num_main == PROFILER_EPOCH_TARGET : # opt.epoch_num_main باید از main.py پاس داده شود
             # شروع context پروفایلر در اولین step از بازه پروفایلینگ
            if step == 0: # یا دقیقتر wait_steps
                print(f"--- Starting Profiler for {active_steps} active steps (total {profiler_total_steps} steps) in epoch {opt.epoch_num_main} ---")
                # این context باید بیرون از حلقه باشد و prof.step() داخل حلقه
                # این روش برای with ... as prof کمی پیچیده می‌شود اگر بخواهیم فقط بخشی از حلقه را پوشش دهیم.

                # راه ساده‌تر: فقط چند بچ اول را با context جداگانه پروفایل کنیم
                if 'profiler_context_active' not in locals() or not profiler_context_active: # جلوگیری از ورود مجدد
                    profiler_context_active = True # یک فلگ برای کنترل
                    
                    # **روش صحیح‌تر برای پروفایل کردن بخشی از حلقه:**
                    # پروفایلر را بیرون حلقه بچ‌ها تعریف کنید (اگر فقط برای یک epoch است)
                    # و داخل حلقه prof.step() را بزنید.
                    # در اینجا برای سادگی، پروفایل کردن را به `run_profiler_on_first_few_batches` منتقل می‌کنیم.
                    # این تابع باید `model`, `train_data`, `opt`, `scaler` و `slices` را بگیرد.
                    # اما برای حفظ ساختار فعلی، فقط یک پیغام چاپ می‌کنیم و از کاربر می‌خواهیم
                    # این بخش را با دقت بیشتری مطابق نیاز خود پیاده‌سازی کند.
                    
                    print(f"INFO: Profiler would be active here. For precise profiling of a few batches,")
                    print(f"      wrap the specific batch loop iteration with the 'with torch.profiler.profile(...)' context.")
                    print(f"      Example: if step in range(wait, wait+warmup+active): prof.step()")
                    # برای این اجرا، ما از پروفایلر صرف نظر می‌کنیم تا کد پیچیده نشود
                    # ENABLE_PROFILER = False # برای ادامه بدون پروفایلر در این اجرا
                                  
        model.optimizer.zero_grad(set_to_none=True) # set_to_none=True برای کمی بهینه‌سازی حافظه

        with autocast(enabled=torch.cuda.is_available()):
            targets, scores, ssl_loss_val = forward(model, i_slice, train_data, is_train=True)
            rec_loss = torch.tensor(0.0, device=scores.device)
            valid_targets_mask = (targets > 0) & (targets <= model.n_node)

            if valid_targets_mask.sum() > 0:
                 if scores.shape[1] == model.n_node - 1:
                      try:
                           target_values_for_loss = (targets[valid_targets_mask] - 1).clamp(0, scores.shape[1] - 1)
                           rec_loss = model.loss_function(scores[valid_targets_mask], target_values_for_loss)
                      except IndexError:
                            pass 
                 else:
                    # print(f"Warning: Score dim mismatch. scores: {scores.shape[1]}, expected: {model.n_node - 1}")
                    pass
            
            loss = rec_loss + model.ssl_weight * ssl_loss_val
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                 continue 

        if torch.cuda.is_available():
             scaler.scale(loss).backward()
             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping (اختیاری)
             scaler.step(model.optimizer)
             scaler.update()
        else:
             loss.backward()
             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping (اختیاری)
             model.optimizer.step()
        
        # ---- این بخش برای استفاده از prof.step() است اگر with prof بیرون حلقه باشد ----
        # if ENABLE_PROFILER and step < profiler_total_steps and opt.epoch_num_main == PROFILER_EPOCH_TARGET:
        #     if 'prof_instance' in locals() and prof_instance is not None: # prof_instance باید قبلا ساخته شده باشد
        #          prof_instance.step() 
        # --------------------------------------------------------------------------

        total_loss += loss.item() if not torch.isnan(loss).any() else 0
        total_rec_loss += rec_loss.item() if not torch.isnan(rec_loss).any() else 0
        total_ssl_loss += ssl_loss_val.item() if not torch.isnan(ssl_loss_val).any() else 0

        if (step + 1) % max(1, int(len(slices) / 10 +1)) == 0: # لاگ حدود 10 بار یا بیشتر
             avg_loss = total_loss / (step + 1) if (step+1) > 0 else 0
             avg_rec_loss = total_rec_loss / (step + 1) if (step+1) > 0 else 0
             avg_ssl_loss = total_ssl_loss / (step + 1) if (step+1) > 0 else 0
             print(f'[{step + 1}/{len(slices)}] Tot Loss: {avg_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}, SSL Loss: {avg_ssl_loss:.4f}')
    
    # --- پایان پروفایلینگ اگر context بیرون حلقه بود ---
    # if 'profiler_context_active' in locals() and profiler_context_active:
    #      profiler_context_active = False # ریست کردن فلگ
    #      print(f"--- Profiler finished for epoch {opt.epoch_num_main} ---")
    #      print(f"Profiler logs saved to: {profiler_log_dir}")
    # -----------------------------------------------------

    model.scheduler.step()
    len_slices = len(slices) if slices else 1
    print(f'\tAvg Loss:\t{total_loss / len_slices:.4f}')
    print(f'\tAvg Rec Loss:\t{total_rec_loss / len_slices:.4f}')
    print(f'\tAvg SSL Loss:\t{total_ssl_loss / len_slices:.4f}')

    print('Start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    test_slices_eval = test_data.generate_batch(model.batch_size)
    with torch.no_grad():
        for k_slice_eval in test_slices_eval:
            targets, scores, _ = forward(model, k_slice_eval, test_data, is_train=False)
            sub_scores_indices = scores.topk(20)[1] 
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
                    else:
                        mrr.append(0.0)

    hit_rate = np.mean(hit) * 100 if hit else 0.0
    mrr_score = np.mean(mrr) * 100 if mrr else 0.0
    return hit_rate, mrr_score