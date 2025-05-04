import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# اضافه کردن autocast و GradScaler اگر قبلاً import نشده‌اند
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
        # x shape: (sequence length, batch size, embed dim) یا (batch_size, sequence length, embed dim)
        # تطبیق با batch_first=True در Transformer
        if x.dim() == 3 and x.size(1) == self.pe.size(0) and x.size(2) == self.pe.size(2):
             # حالت (batch_size, sequence length, embed dim)
             x = x + self.pe[:x.size(1), :].squeeze(1) # self.pe shape (max_len, 1, d_model) -> (seq_len, d_model)
        elif x.dim() == 3 and x.size(0) == self.pe.size(0):
             # حالت (sequence length, batch size, embed dim)
             x = x + self.pe[:x.size(0), :]
        else:
            # Handle cases where dimensions might not match perfectly during inference or padding
            # Find the sequence length dimension dynamically
            seq_len_dim_index = -1
            if x.dim() == 3:
                if x.size(1) <= self.pe.size(0): # Check batch_first=True case
                    seq_len_dim_index = 1
                elif x.size(0) <= self.pe.size(0): # Check batch_first=False case
                    seq_len_dim_index = 0

            if seq_len_dim_index != -1:
                seq_len = x.size(seq_len_dim_index)
                if seq_len_dim_index == 1: # batch_first = True
                     pe_to_add = self.pe[:seq_len, :].squeeze(1) # Shape: (seq_len, d_model)
                     # Ensure broadcasting works: pe_to_add might need unsqueeze(0) if x is (batch, seq, dim)
                     if x.dim() == 3: pe_to_add = pe_to_add.unsqueeze(0) # Shape: (1, seq_len, d_model)
                else: # batch_first = False
                     pe_to_add = self.pe[:seq_len, :] # Shape: (seq_len, 1, d_model)

                # Check if dimensions match before adding
                if x.shape == pe_to_add.shape or (x.dim() == 3 and pe_to_add.dim() == 3 and x.shape[2] == pe_to_add.shape[2]):
                     # Allow broadcasting if batch or sequence dim differs but embed_dim matches
                     try:
                         x = x + pe_to_add
                     except RuntimeError as e:
                         print(f"Warning: PositionalEncoding shape mismatch. x: {x.shape}, pe: {pe_to_add.shape}. Error: {e}. Skipping addition.")

                else:
                     print(f"Warning: PositionalEncoding shape mismatch. x: {x.shape}, pe: {pe_to_add.shape}. Skipping addition.")

            else:
                 print(f"Warning: Could not determine sequence length dimension or sequence too long for PositionalEncoding. x shape: {x.shape}, max_len: {self.pe.size(0)}. Skipping addition.")

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
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        # Ensure A and hidden are on the same device
        device = hidden.device
        A = A.to(device)

        # Debug shapes
        # print("GNNCell - A shape:", A.shape) # Expected: (batch, max_n_node, 2 * max_n_node)
        # print("GNNCell - hidden shape:", hidden.shape) # Expected: (batch, max_n_node, hidden_size)

        # Check for NaN/Inf in inputs
        if torch.isnan(A).any() or torch.isinf(A).any():
             print("Warning: NaN/Inf detected in GNN input A.")
             A = torch.nan_to_num(A)
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
             print("Warning: NaN/Inf detected in GNN input hidden.")
             hidden = torch.nan_to_num(hidden)


        # Check matrix multiplication dimensions carefully
        # A[:, :, :A.shape[1]] shape: (batch, max_n_node, max_n_node)
        # self.linear_edge_in(hidden) shape: (batch, max_n_node, hidden_size)
        # Result should be: (batch, max_n_node, hidden_size)

        try:
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        except RuntimeError as e:
             print("Error during GNN matmul:")
             print("A shape:", A.shape)
             print("hidden shape:", hidden.shape)
             print("linear_edge_in(hidden) shape:", self.linear_edge_in(hidden).shape)
             print("A[:, :, :A.shape[1]] shape:", A[:, :, :A.shape[1]].shape)
             print("A[:, :, A.shape[1]: 2 * A.shape[1]] shape:", A[:, :, A.shape[1]: 2 * A.shape[1]].shape)
             raise e


        inputs = torch.cat([input_in, input_out], 2) # shape: (batch, max_n_node, 2 * hidden_size)
        gi = F.linear(inputs, self.w_ih, self.b_ih) # W_ih: (gate_size, input_size) -> (3h, 2h)
        gh = F.linear(hidden, self.w_hh, self.b_hh) # W_hh: (gate_size, hidden_size) -> (3h, h)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        # Check for NaN/Inf in output
        if torch.isnan(hy).any() or torch.isinf(hy).any():
             print("Warning: NaN/Inf detected in GNN output hy.")
             hy = torch.nan_to_num(hy)

        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

# -------------- 3. کلاس SessionGraph (تغییر یافته برای SSL) --------------
class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0) # اضافه کردن padding_idx=0
        self.gnn = GNN(self.hidden_size, step=opt.step)

        # --- پارامترهای SSL ---
        self.ssl_weight = opt.ssl_weight
        self.ssl_temp = opt.ssl_temp
        self.ssl_dropout_rate = opt.ssl_dropout_rate
        # --------------------

        # --- Transformer Layers ---
        nhead = getattr(opt, 'nhead', 2) # getattr برای ایمنی بیشتر
        num_encoder_layers = getattr(opt, 'nlayers', 2)
        dim_feedforward = getattr(opt, 'ff_hidden', 256)
        dropout = getattr(opt, 'dropout', 0.1)

        self.pos_encoder = PositionalEncoding(self.hidden_size, dropout)
        # استفاده از batch_first=True برای سازگاری با ورودی (batch, seq, feature)
        encoder_layers = TransformerEncoderLayer(self.hidden_size, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        # -------------------------

        # --- لایه های محاسبه امتیاز ---
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # ---------------------------

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                # مقداردهی اولیه Xavier برای لایه‌های Transformer و Linear
                nn.init.xavier_uniform_(weight)
            else:
                # مقداردهی اولیه برای bias ها و embedding ها (اگر تک بعدی باشند)
                nn.init.uniform_(weight, -stdv, stdv)
        # اطمینان از مقداردهی اولیه embedding (می‌تواند متفاوت باشد)
        nn.init.uniform_(self.embedding.weight, -stdv, stdv)
         # اطمینان از اینکه پدینگ صفر است
        if self.embedding.padding_idx is not None:
             with torch.no_grad():
                  self.embedding.weight[self.embedding.padding_idx].fill_(0)


    def compute_scores(self, hidden_transformer_output, mask):
        # hidden_transformer_output shape: (batch_size, seq_length, hidden_size)
        # mask shape: (batch_size, seq_length)

        # Debug shape
        # print("compute_scores - hidden_transformer_output shape:", hidden_transformer_output.shape)
        # print("compute_scores - mask shape:", mask.shape)

        # پیدا کردن اندیس آخرین آیتم معتبر در هر سشن
        # sequence_lengths = torch.sum(mask, 1)
        # Ensure sequence_lengths are valid indices (>= 0)
        # gather_index = (sequence_lengths - 1).clamp(min=0) # Clamp to avoid -1 index for empty sequences

        # Use torch.gather for safe indexing, handles variable lengths
        # We need indices for the gather operation: (batch_size, 1, hidden_size)
        # gather_index shape: (batch_size) -> (batch_size, 1) -> (batch_size, 1, 1) -> (batch_size, 1, hidden_size)
        # gather_index = gather_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hidden_transformer_output.size(-1))

        # ht = torch.gather(hidden_transformer_output, 1, gather_index).squeeze(1) # batch_size x hidden_size

        # روش ساده‌تر و رایج‌تر با فرض اینکه mask معتبر است:
        try:
            sequence_lengths = torch.sum(mask.float(), 1).long()
             # Handle cases where sequence length might be 0 after processing
            batch_indices = torch.arange(mask.shape[0], device=mask.device)
            valid_lengths_mask = sequence_lengths > 0
            gather_indices = (sequence_lengths[valid_lengths_mask] - 1)

            ht = torch.zeros(mask.shape[0], hidden_transformer_output.size(-1), device=hidden_transformer_output.device)
            if gather_indices.numel() > 0: # Only gather if there are valid sequences
                 ht[valid_lengths_mask] = hidden_transformer_output[batch_indices[valid_lengths_mask], gather_indices]

        except IndexError as e:
             print("Error gathering last item embedding:")
             print("hidden_transformer_output shape:", hidden_transformer_output.shape)
             print("mask shape:", mask.shape)
             print("sequence_lengths:", sequence_lengths)
             print("gather_indices:", gather_indices)
             print("valid_lengths_mask:", valid_lengths_mask)
             print("batch_indices[valid_lengths_mask]:", batch_indices[valid_lengths_mask])
             raise e


        # --- محاسبه session representation ---
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden_transformer_output)

        # Apply mask before softmax for attention alpha
        # We want to attend only to valid items in the sequence
        # Add a large negative number to masked positions before sigmoid/softmax
        mask_expanded = mask.unsqueeze(-1).float() # (b, s, 1)
        # Prevent masking everything if a sequence is full of padding (though should not happen with proper data)
        # safe_mask = mask_expanded + (1.0 - mask_expanded) * -1e9 # Assign large negative to 0s in mask

        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)) # (b, s, 1)
        # Mask before softmax
        alpha_logits_masked = alpha_logits.masked_fill(mask_expanded == 0, -float('inf'))
        alpha = F.softmax(alpha_logits_masked, dim=1) # (b, s, 1)


        # Apply mask again during weighted sum
        a = torch.sum(alpha * hidden_transformer_output * mask_expanded, 1) # (b, d) - Global preference

        # ترکیب با last item embedding (ht)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1)) # (b, d)

        # --- Target Attention ---
        # اطمینان از اینکه embedding ها روی همان device هستند
        b = self.embedding.weight[1:].to(a.device) # n_nodes-1 x d

        # hidden_masked = hidden_transformer_output * mask_expanded # (b, s, d)
        # qt = self.linear_t(hidden_masked) # (b, s, d)

        # Mask the input to linear_t
        hidden_masked = hidden_transformer_output * mask_expanded # b, s, d
        qt = self.linear_t(hidden_masked) # b, s, d

        # Masking before softmax in beta calculation
        # Need to mask based on the sequence mask
        # beta_logits = torch.matmul(b, qt.transpose(1, 2)) # (b, n, s)
        beta_logits = b @ qt.transpose(1, 2) # (b, n-1, s)

        # Create mask for beta softmax: (b, 1, s)
        beta_mask = mask.unsqueeze(1) # (b, 1, s)
        beta_logits_masked = beta_logits.masked_fill(beta_mask == 0, -float('inf'))
        beta = F.softmax(beta_logits_masked, dim=-1) # (b, n-1, s)


        # Apply sequence mask again to qt before final matmul
        target = beta @ (qt * mask_expanded.transpose(1,2)) # (b, n-1, d)

        # --- ترکیب نهایی و محاسبه امتیاز ---
        final_representation = a.unsqueeze(1) + target # (b, n-1, d) Use broadcasting
        # final_representation = a.view(ht.shape[0], 1, ht.shape[1]) + target # Old way

        # محاسبه امتیاز نهایی (dot product)
        # Ensure dimensions match for batch matrix multiplication or element-wise product sum
        # final_representation: (b, n-1, d)
        # b: (n-1, d) -> unsqueeze(0) -> (1, n-1, d)
        scores = torch.sum(final_representation * b.unsqueeze(0), dim=-1) # (b, n-1)

        # scores = torch.matmul(final_representation, b.transpose(0, 1)) # (b, n-1) - Alternative if shapes align

        return scores


    # -------------- تابع محاسبه زیان SSL (InfoNCE) --------------
    def calculate_ssl_loss(self, emb1, emb2, temperature):
        # emb1, emb2: shape (batch_size, hidden_size) - بازنمایی‌های دو نمای افزوده
        batch_size = emb1.shape[0]
        device = emb1.device

        # نرمال سازی L2 برای پایداری
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        # محاسبه ماتریس شباهت بین تمام زوج‌های ممکن در بچ
        sim_matrix_12 = torch.matmul(emb1, emb2.t()) / temperature # (batch_size, batch_size)
        sim_matrix_21 = torch.matmul(emb2, emb1.t()) / temperature # (batch_size, batch_size)

        # نمونه‌های مثبت در قطر اصلی قرار دارند
        # زیان برای emb1 -> emb2
        logits_12 = sim_matrix_12 - torch.diag(torch.diag(sim_matrix_12)) # صفر کردن قطر اصلی برای محاسبه زیان
        # استفاده از log_softmax برای پایداری عددی
        log_softmax_12 = F.log_softmax(sim_matrix_12, dim=1)
        loss_12 = -torch.diag(log_softmax_12) # گرفتن عناصر قطر اصلی (نمونه‌های مثبت)

        # زیان برای emb2 -> emb1
        logits_21 = sim_matrix_21 - torch.diag(torch.diag(sim_matrix_21))
        log_softmax_21 = F.log_softmax(sim_matrix_21, dim=1)
        loss_21 = -torch.diag(log_softmax_21)

        # میانگین دو زیان
        ssl_loss = (loss_12.mean() + loss_21.mean()) / 2.0
        return ssl_loss
    # ---------------------------------------------------------

# تابع forward اصلی مدل GNN (بدون تغییر، فقط نامگذاری برای وضوح)
# def gnn_forward(model, inputs, A):
#     hidden = model.embedding(inputs)
#     hidden = model.gnn(A, hidden)
#     return hidden


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


# -------------- تابع forward کلی (تغییر یافته برای SSL) --------------
def forward(model, i, data, is_train=True):
    # گرفتن داده های بچ
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    # A باید از قبل float باشد، تبدیل به تنسور
    A = trans_to_cuda(torch.tensor(np.array(A), dtype=torch.float32)) # تعیین dtype
    mask = trans_to_cuda(torch.Tensor(mask).long())
    targets = trans_to_cuda(torch.Tensor(targets).long()) # انتقال targets به CUDA

    # 1. Embedding Layer
    hidden = model.embedding(items) # (batch, max_n_node, hidden_size)

    # Check for NaN/Inf after embedding
    if torch.isnan(hidden).any() or torch.isinf(hidden).any():
        print(f"Warning: NaN/Inf detected after embedding layer for batch slice {i}.")
        hidden = torch.nan_to_num(hidden)


    # 2. GNN Layer
    hidden_gnn = model.gnn(A, hidden) # (batch, max_n_node, hidden_size)

    # Check for NaN/Inf after GNN
    if torch.isnan(hidden_gnn).any() or torch.isinf(hidden_gnn).any():
        print(f"Warning: NaN/Inf detected after GNN layer for batch slice {i}.")
        hidden_gnn = torch.nan_to_num(hidden_gnn)


    # 3. استخراج بازنمایی‌های توالی برای مسیر اصلی توصیه
    # alias_inputs: (batch, seq_len) -> map session items to nodes in the graph 'items'
    # hidden_gnn: (batch, max_n_node, hidden_size)
    # We need: (batch, seq_len, hidden_size)
    # Use gather or direct indexing if possible, handle padding index (0) if necessary
    batch_size, seq_len = alias_inputs.shape
    hidden_size = hidden_gnn.shape[-1]
    # Create batch indices for gather
    batch_indices = torch.arange(batch_size, device=alias_inputs.device).unsqueeze(1).expand(-1, seq_len) # (b, s)
    # Gather embeddings based on alias_inputs
    # alias_inputs needs to be adjusted if it contains indices out of bounds for hidden_gnn's node dimension
    # Assuming alias_inputs contains valid indices relative to the nodes in 'items' which correspond to 'hidden_gnn'
    try:
         # Clamp alias_inputs to be within valid range [0, max_n_node-1]
         # max_n_node = hidden_gnn.shape[1]
         # clamped_alias = alias_inputs.clamp(0, max_n_node - 1)
         # seq_hidden_gnn = hidden_gnn[batch_indices, clamped_alias] # Shape: (b, s, d)

         # Safer way using gather if direct indexing fails
         # Ensure alias_inputs is (B, S, 1) and expanded
         alias_expanded = alias_inputs.unsqueeze(-1).expand(-1, -1, hidden_size) # (b, s, d)
         # We need to gather along dimension 1 (the node dimension)
         seq_hidden_gnn = torch.gather(hidden_gnn, 1, alias_expanded) # Potential shape mismatch if alias_inputs has wrong indices


    except IndexError as e:
         print("Error gathering sequence embeddings:")
         print("hidden_gnn shape:", hidden_gnn.shape)
         print("alias_inputs shape:", alias_inputs.shape)
         # print("clamped_alias shape:", clamped_alias.shape)
         print("batch_indices shape:", batch_indices.shape)
         print(f"Error: {e}")
         # Handle error, maybe return zero tensors or raise
         # For now, let's try to continue with zeros, but this indicates a problem
         print("Warning: Falling back to zero tensor due to gathering error.")
         seq_hidden_gnn = torch.zeros(batch_size, seq_len, hidden_size, device=model.embedding.weight.device)

    # --- مسیر اصلی توصیه ---
    # 4. Positional Encoding
    seq_hidden_pos = model.pos_encoder(seq_hidden_gnn) # Input/Output: (batch, seq, feature)

    # 5. Transformer Encoder
    src_key_padding_mask = (mask == 0) # (batch_size, seq_len) - True where padded
    hidden_transformer_output = model.transformer_encoder(
        src=seq_hidden_pos,
        # mask=None, # TransformerEncoder doesn't take a causal mask by default
        src_key_padding_mask=src_key_padding_mask
    ) # Output: (batch, seq, feature)

    # 6. محاسبه امتیازات توصیه
    scores = model.compute_scores(hidden_transformer_output, mask) # Output: (b, n_node-1)

    # --- مسیر SSL (فقط در حالت آموزش) ---
    ssl_loss = torch.tensor(0.0, device=scores.device) # مقدار پیش‌فرض
    if is_train and model.ssl_weight > 0:
        # 7. ایجاد دو نمای افزوده با Dropout روی خروجی GNN
        # استخراج بازنمایی آخرین آیتم از خروجی GNN برای SSL
        try:
            sequence_lengths = torch.sum(mask.float(), 1).long()
            batch_indices_ssl = torch.arange(batch_size, device=alias_inputs.device)
            valid_lengths_mask_ssl = sequence_lengths > 0
            last_item_node_indices = torch.zeros(batch_size, dtype=torch.long, device=alias_inputs.device)

            if valid_lengths_mask_ssl.any():
                 # Get the alias index of the last valid item in the original sequence
                 last_valid_seq_indices = sequence_lengths[valid_lengths_mask_ssl] - 1
                 # Get the corresponding node index from alias_inputs
                 last_item_node_indices[valid_lengths_mask_ssl] = alias_inputs[batch_indices_ssl[valid_lengths_mask_ssl], last_valid_seq_indices]


            # استخراج embedding مربوطه از hidden_gnn
            # Shape: (batch, max_n_node, hidden_size) -> (batch, hidden_size)
            ssl_base_emb = torch.zeros(batch_size, hidden_size, device=hidden_gnn.device)
            if valid_lengths_mask_ssl.any():
                  ssl_base_emb[valid_lengths_mask_ssl] = hidden_gnn[batch_indices_ssl[valid_lengths_mask_ssl], last_item_node_indices[valid_lengths_mask_ssl]]


            # اعمال Dropout دو بار برای ایجاد نماها
            ssl_emb1 = F.dropout(ssl_base_emb, p=model.ssl_dropout_rate, training=True)
            ssl_emb2 = F.dropout(ssl_base_emb, p=model.ssl_dropout_rate, training=True)

            # 8. محاسبه زیان SSL
            ssl_loss = model.calculate_ssl_loss(ssl_emb1, ssl_emb2, model.ssl_temp)

            # Check for NaN/Inf in SSL loss
            if torch.isnan(ssl_loss).any() or torch.isinf(ssl_loss).any():
                 print(f"Warning: NaN/Inf detected in SSL loss for batch slice {i}. Setting to 0.")
                 ssl_loss = torch.tensor(0.0, device=scores.device)


        except Exception as e:
             print(f"Error during SSL calculation for batch slice {i}: {e}")
             # print("Shapes: hidden_gnn {hidden_gnn.shape}, alias_inputs {alias_inputs.shape}, mask {mask.shape}")
             ssl_loss = torch.tensor(0.0, device=scores.device) # خطا را نادیده بگیر و ادامه بده

    # Return targets, scores, and ssl_loss
    # Ensure targets are on the correct device
    return targets.to(scores.device), scores, ssl_loss
# ---------------------------------------------------------------------


# -------------- تابع آموزش و تست (تغییر یافته برای SSL) --------------
# scaler = GradScaler() # scaler را می‌توان به صورت گلوبال تعریف کرد یا در train_test

def train_test(model, train_data, test_data, opt): # اضافه کردن opt
    # scaler را در ابتدای تابع train_test ایجاد کنید
    # enabled=torch.cuda.is_available() مهم است تا در حالت CPU خطا ندهد
    scaler = GradScaler(enabled=torch.cuda.is_available())

    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    total_rec_loss = 0.0
    total_ssl_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for step, i in enumerate(slices): # استفاده از enumerate برای لاگ بهتر
        model.optimizer.zero_grad()

        # استفاده از autocast فقط در صورت فعال بودن CUDA
        with autocast(enabled=torch.cuda.is_available()):
            # فراخوانی forward با is_train=True
            targets, scores, ssl_loss = forward(model, i, train_data, is_train=True)

            # محاسبه زیان توصیه (مطمئن شوید targets معتبر هستند)
            # targets باید 1-based باشند برای CrossEntropyLoss اگر خروجی 0-based است
            # scores shape: (b, n_node-1), targets shape: (b)
            # target - 1 باید در محدوده [0, n_node-2] باشد
            valid_targets_mask = (targets > 0) & (targets <= model.n_node) # فقط تارگت‌های معتبر
            if valid_targets_mask.sum() == 0:
                 print(f"Warning: No valid targets in batch slice {i}. Skipping loss calculation.")
                 rec_loss = torch.tensor(0.0, device=scores.device)
            else:
                 # اطمینان از اینکه scores و targets[valid_targets_mask]-1 سازگار هستند
                 # scores باید برای نودهای 1 تا n_node باشد -> اندیس 0 تا n_node-2
                 if scores.shape[1] != model.n_node - 1:
                      print(f"Warning: Score dimension mismatch. scores: {scores.shape[1]}, expected: {model.n_node - 1}")
                      # شاید نیاز به تنظیم مدل یا داده باشد، فعلا رد می‌شویم
                      rec_loss = torch.tensor(0.0, device=scores.device)
                 else:
                      try:
                           rec_loss = model.loss_function(scores[valid_targets_mask], targets[valid_targets_mask] - 1)
                      except IndexError as e:
                            print(f"Error in CrossEntropyLoss for batch slice {i}:")
                            print("scores shape:", scores.shape)
                            print("targets shape:", targets.shape)
                            print("valid_targets_mask shape:", valid_targets_mask.shape)
                            print("valid targets:", targets[valid_targets_mask])
                            print("max target-1:", (targets[valid_targets_mask] - 1).max())
                            print("min target-1:", (targets[valid_targets_mask] - 1).min())
                            print(f"Error: {e}")
                            rec_loss = torch.tensor(0.0, device=scores.device) # رد شدن از بچ مشکل‌دار


            # ترکیب زیان ها
            loss = rec_loss + model.ssl_weight * ssl_loss

            # Check for NaN/Inf before backward
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                 print(f"Warning: NaN/Inf detected in total loss for batch slice {i} before backward. Skipping batch.")
                 continue # پرش از این بچ

        # Backward pass با scaler
        # scaler.scale(loss).backward()
        # scaler.step(model.optimizer)
        # scaler.update()

        # کنترل دستی برای جلوگیری از خطای scaler در CPU
        if torch.cuda.is_available():
             scaler.scale(loss).backward()
              # Gradient clipping (optional but recommended)
             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
             scaler.step(model.optimizer)
             scaler.update()
        else:
             loss.backward()
             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clipping for CPU too
             model.optimizer.step()


        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_ssl_loss += ssl_loss.item() # ssl_loss خودش یک تنسور اسکالر است

        if (step + 1) % int(len(slices) / 5 + 1) == 0:
             avg_loss = total_loss / (step + 1)
             avg_rec_loss = total_rec_loss / (step + 1)
             avg_ssl_loss = total_ssl_loss / (step + 1) # میانگین زیان ssl

             print('[%d/%d] Tot Loss: %.4f, Rec Loss: %.4f, SSL Loss: %.4f' %
                   (step + 1, len(slices), avg_loss, avg_rec_loss, avg_ssl_loss))

    # به روز رسانی نرخ یادگیری
    model.scheduler.step()


    print('\tAvg Loss:\t%.4f' % (total_loss / len(slices)))
    print('\tAvg Rec Loss:\t%.4f' % (total_rec_loss / len(slices)))
    print('\tAvg SSL Loss:\t%.4f' % (total_ssl_loss / len(slices)))


    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    with torch.no_grad(): # غیرفعال کردن محاسبه گرادیان در هنگام تست
        for i in slices:
            # فراخوانی forward با is_train=False، فقط به scores نیاز داریم
            targets, scores, _ = forward(model, i, test_data, is_train=False)
            sub_scores = scores.topk(20)[1] # گرفتن اندیس ۲۰ آیتم برتر
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()

            targets_cpu = trans_to_cpu(targets).detach().numpy()


            # محاسبه Hit Rate و MRR
            for score_indices, target in zip(sub_scores, targets_cpu):
                 # score_indices از 0 تا n_node-2 است
                 # target از 1 تا n_node است
                 if target > 0: # فقط برای تارگت های معتبر
                    target_zero_based = target - 1
                    # بررسی آیا target_zero_based در score_indices وجود دارد
                    hit.append(np.isin(target_zero_based, score_indices))
                    # پیدا کردن رتبه target_zero_based
                    where_result = np.where(score_indices == target_zero_based)[0]
                    if len(where_result) == 0:
                        mrr.append(0)
                    else:
                        rank = where_result[0] + 1
                        mrr.append(1 / rank)

    hit = np.mean(hit) * 100 if hit else 0
    mrr = np.mean(mrr) * 100 if mrr else 0
    return hit, mrr
# ---------------------------------------------------------------------