import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import math
import copy # برای TargetAware

# -------------- 1. کلاس PositionalEncoding (از کد اصلی شما) --------------
class PositionalEncoding(nn.Module):
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

    def forward(self, x): # x: (B, L, D)
        pe_to_add = self.pe[:x.size(1), :].squeeze(1).unsqueeze(0)
        try:
            x = x + pe_to_add
        except RuntimeError:
             seq_len = x.size(1)
             if seq_len <= self.pe.size(0):
                 pe_to_add_resized = self.pe[:seq_len, :].squeeze(1).unsqueeze(0)
                 x = x + pe_to_add_resized
        return self.dropout(x)


# -------------- 2. TargetAwareEncoderLayer (از کد اصلی شما، با کمی تغییر احتمالی) --------------
# این لایه باید ورودی src (B,L,D) و candidate_embeddings_global (N_all_items, D) را دریافت کند
class TargetAwareEncoderLayer(nn.Module):
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

    def forward(self, src, candidate_embeddings_global, src_key_padding_mask=None, src_mask=None):
        # src: (B, L, D)
        # candidate_embeddings_global: (N_all_items, D)
        q_sc = self.W_sc_q(src)
        k_sc = self.W_sc_k(candidate_embeddings_global)
        v_sc = self.W_sc_v(candidate_embeddings_global)
        attn_score_sc = torch.matmul(q_sc, k_sc.transpose(-2, -1)) / math.sqrt(q_sc.size(-1)) # (B,L,N)
        attn_weights_sc = F.softmax(attn_score_sc, dim=-1)
        context_from_candidates = torch.matmul(attn_weights_sc, v_sc) # (B,L,D)
        src_enhanced = src + context_from_candidates
        sa_output, _ = self.self_attn(src_enhanced, src_enhanced, src_enhanced,
                                      key_padding_mask=src_key_padding_mask,
                                      attn_mask=src_mask)
        out1 = src + self.dropout1(sa_output) # اتصال با src اصلی (مانند کد شما)
        out1 = self.norm1(out1)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(out1))))
        out2 = out1 + self.dropout2(ff_output)
        out2 = self.norm2(out2)
        return out2

# -------------- 3. TargetAwareTransformerEncoder (از کد اصلی شما) --------------
class TargetAwareTransformerEncoder(nn.Module):
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


class SessionGraph(nn.Module):
    def __init__(self, opt, n_node): # از opt برای پارامترها استفاده می‌کنیم
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.nonhybrid = opt.nonhybrid
        self.ssl_weight = opt.ssl_weight
        self.ssl_temp = opt.ssl_temp
        self.ssl_dropout_rate = opt.ssl_dropout_rate


        self.embedding = nn.Embedding(n_node, self.hidden_size, padding_idx=0)
        self.gnn = GatedGraphConv(out_channels=self.hidden_size, num_layers=opt.step) # step از opt

        self.pos_encoder = PositionalEncoding(self.hidden_size, dropout=opt.dropout) # dropout از opt

        ta_encoder_layer = TargetAwareEncoderLayer(
            d_model=self.hidden_size,
            nhead=opt.nhead, # nhead از opt
            dim_feedforward=opt.ff_hidden, # ff_hidden از opt
            dropout=opt.dropout # dropout از opt
        )
        self.transformer_encoder = TargetAwareTransformerEncoder(
            encoder_layer=ta_encoder_layer,
            num_layers=opt.nlayers, # nlayers از opt
            norm=nn.LayerNorm(self.hidden_size)
        )

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        # Optimizer و Scheduler در main.py تعریف می‌شوند

    # reset_parameters از کد اصلی شما (اگر لازم است)
    # def reset_parameters(self): ...

    def map_gnn_to_sequence(self, gnn_output_flat, pyg_batch, original_sequence_ids_padded, current_batch_size):
        """
        **این تابع چالش‌برانگیزترین بخش است و نیاز به پیاده‌سازی دقیق و کارآمد دارد.**
        هدف: برای هر توالی در بچ (original_sequence_ids_padded)، نمایش نودهای GNN متناظر را استخراج کند.
        gnn_output_flat: (num_total_nodes_in_batch, D) خروجی GNN
        pyg_batch.x: (num_total_nodes_in_batch,) IDهای اصلی نودهای یکتا در GNN
        pyg_batch.batch: (num_total_nodes_in_batch,) نگاشت هر نود GNN به گرافش در بچ
        original_sequence_ids_padded: (B, L_padded) IDهای اصلی آیتم‌ها در توالی‌ها
        current_batch_size: B واقعی (ممکن است کوچکتر از B اولیه باشد اگر جلسات خالی حذف شده باشند)

        خروجی باید: (B, L_padded, D)
        """
        device = gnn_output_flat.device
        padded_seq_len = original_sequence_ids_padded.size(1)
        output_sequence_embeddings = torch.zeros(current_batch_size, padded_seq_len, self.hidden_size, device=device)
        
        # 1. ایجاد یک نگاشت از ID اصلی آیتم به اندیس آن در gnn_output_flat برای بچ فعلی
        # این کارآمد نیست اگر در هر بچ تکرار شود. باید راه بهتری پیدا کرد.
        # pyg_batch.x شامل IDهای اصلی نودهای منحصربه‌فرد در کل بچ است.
        # gnn_output_flat نمایش متناظر این نودها است.

        # یک راه (ممکن است کند باشد برای IDهای زیاد):
        # برای هر گراف در بچ
        processed_graphs_in_pyg_batch = 0
        for i in range(current_batch_size): # حلقه روی بچ اصلی (قبل از فیلتر جلسات خالی)
            # پیدا کردن نودهای GNN مربوط به جلسه i-ام (اگر جلسه معتبر بوده و در pyg_batch هست)
            # این بخش نیاز به نگاشت دقیق بین اندیس i بچ اصلی و اندیس گراف در pyg_batch دارد
            # اگر pyg_batch از قبل شامل تمام جلسات (حتی خالی‌ها به صورت placeholder) باشد، ساده‌تر است.
            # در اینجا فرض می‌کنیم pyg_batch فقط شامل گراف‌های معتبر است.
            # این بخش نیاز به بازبینی جدی دارد که چگونه اندیس i بچ اصلی به گراف در pyg_batch نگاشت می‌شود.
            
            # راه حل ساده‌تر (اما با فرض): اگر بتوانیم یک نگاشت مستقیم از ID به embedding GNN داشته باشیم
            # که برای کل بچ یکبار ساخته شود.
            pass # این تابع نیاز به پیاده‌سازی بسیار دقیق دارد.

        # **روش جایگزین و ساده‌تر (اما با تغییر منطق):**
        # مستقیماً original_sequence_ids_padded را امبد کرده و از آن به عنوان ورودی Transformer استفاده کنید.
        # سپس خروجی GNN را برای اهداف دیگر (SSL یا ترکیب با ht/a) به کار ببرید.
        # این همان کاری است که در نسخه قبلی PyG انجام دادیم.
        # اگر این کار را بکنید، این تابع map_gnn_to_sequence لازم نیست.
        # در اینجا، برای حفظ ایده اصلی، فرض می‌کنیم این تابع به نحوی پیاده‌سازی شده
        # یا اینکه ورودی Transformer مستقیماً از امبدینگ توالی‌های اصلی می‌آید.

        # برای این مثال، فرض می‌کنیم ورودی Transformer همان امبدینگ توالی‌های اصلی است:
        return self.embedding(original_sequence_ids_padded)


    def forward(self, pyg_batch, original_sequence_ids_padded, attention_masks_transformer, original_sequence_lens, is_train=True):
        # pyg_batch: خروجی Batch.from_data_list (فقط شامل گراف‌های معتبر)
        # original_sequence_ids_padded: (B_all, L_padded) IDهای اصلی، پد شده
        # attention_masks_transformer: (B_all, L_padded) ماسک بولی برای Transformer
        # original_sequence_lens: (B_all) طول اصلی هر توالی
        # B_all: اندازه بچ اولیه قبل از حذف جلسات خالی

        current_batch_size = original_sequence_ids_padded.size(0)
        device = self.embedding.weight.device # گرفتن دستگاه از مدل

        # 1. بخش GNN
        gnn_output_flat = None
        if pyg_batch is not None and pyg_batch.x is not None and pyg_batch.x.numel() > 0 :
            # pyg_batch.x باید IDهای اصلی آیتم‌ها برای embedding باشد
            node_features_embedded = self.embedding(pyg_batch.x) # شکل (num_total_nodes, D)
            gnn_output_flat = self.gnn(node_features_embedded, pyg_batch.edge_index) # (num_total_nodes, D)
        else: # اگر بچ خالی از گراف معتبر بود
            # ایجاد یک خروجی GNN خالی یا صفر با ابعاد مناسب (اگر لازم باشد)
            # یا در ادامه کد مدیریت شود.
            pass


        # 2. آماده‌سازی ورودی برای TargetAwareTransformerEncoder
        # **چالش اصلی در این بخش است**
        # رویکرد ۱: استفاده از خروجی GNN برای ساخت ورودی Transformer (پیچیده)
        # transformer_input_features = self.map_gnn_to_sequence(gnn_output_flat, pyg_batch, original_sequence_ids_padded, current_batch_size)
        # رویکرد ۲ (ساده‌تر، مانند قبل): استفاده از امبدینگ مستقیم توالی‌های اصلی
        transformer_input_features = self.embedding(original_sequence_ids_padded) # (B, L_padded, D)

        transformer_input_pos = self.pos_encoder(transformer_input_features)

        # 3. بخش TargetAwareTransformerEncoder
        # src_key_padding_mask باید True باشد برای آیتم‌هایی که باید نادیده گرفته شوند (پدینگ)
        # attention_masks_transformer از collate_fn می‌آید (True برای آیتم‌های معتبر)
        src_key_padding_mask_transformer = ~attention_masks_transformer

        # candidate_embeddings_global باید امبدینگ تمام آیتم‌های ممکن در دیتاست باشد
        all_item_ids = torch.arange(1, self.n_node, device=device) # آیتم‌ها از 1 تا n_node-1 (0 پدینگ است)
        candidate_embeddings_global = self.embedding(all_item_ids) # (n_node-1, D)

        transformer_output = self.transformer_encoder(
            src=transformer_input_pos,
            candidate_embeddings_global=candidate_embeddings_global,
            src_key_padding_mask=src_key_padding_mask_transformer
        )
        # transformer_output: (B, L_padded, D)

        # 4. محاسبه امتیازات نهایی (با استفاده از منطق compute_scores کد اصلی شما)
        # ورودی‌ها: خروجی Transformer و ماسک آن (و طول‌های اصلی برای ht)
        scores = self._compute_scores_from_transformer(
            transformer_output,
            attention_masks_transformer, # ماسک بولی (True برای معتبر)
            original_sequence_lens # برای محاسبه ht
        )

        # 5. بخش SSL (نیاز به بازنویسی دقیق دارد)
        ssl_loss_value = torch.tensor(0.0, device=scores.device)
        if is_train and self.ssl_weight > 0 and gnn_output_flat is not None:
            # از gnn_output_flat و pyg_batch.batch برای SSL استفاده کنید
            # این بخش باید با دقت بر اساس منطق SSL اصلی شما و با داده‌های PyG بازنویسی شود.
            # مثال ساده با pooling (ممکن است مناسب نباشد):
            if pyg_batch.num_graphs > 0 and gnn_output_flat.numel() > 0:
                # فقط برای جلساتی که بیش از یک نود دارند
                # این بخش باید با دقت بیشتری نوشته شود
                # pooled_for_ssl = global_mean_pool(gnn_output_flat, pyg_batch.batch)
                # if pooled_for_ssl.size(0) >= 2: # نیاز به حداقل دو نمونه
                #     view1 = self.dropout(pooled_for_ssl)
                #     view2 = self.dropout(pooled_for_ssl)
                #     # یک پیاده‌سازی ساده از InfoNCE (باید تکمیل شود)
                #     # ssl_loss_value = self.calculate_infonce_loss(view1, view2, self.ssl_temp)
                pass # فعلاً SSL را غیرفعال نگه می‌داریم

            ssl_loss_value = ssl_loss_value * self.ssl_weight


        if is_train:
            return scores, ssl_loss_value
        else:
            return scores

    def _compute_scores_from_transformer(self, transformer_output, attention_mask_transformer, original_sequence_lens):
        """
        این متد امتیازات را بر اساس خروجی Transformer محاسبه می‌کند (شبیه به compute_scores کد اصلی شما).
        attention_mask_transformer: ماسک بولی (True برای آیتم‌های معتبر)
        original_sequence_lens: تنسور طول‌های اصلی هر توالی در بچ
        """
        device = transformer_output.device
        batch_size = transformer_output.size(0)

        # ماسک عددی برای محاسبات sum و masked_fill
        mask_float_transformer = attention_mask_transformer.float()

        # محاسبه ht (آخرین آیتم معتبر از خروجی Transformer)
        ht_transformer = torch.zeros(batch_size, self.hidden_size, device=device)
        # original_sequence_lens از collate_fn می‌آید
        valid_lengths_mask = original_sequence_lens > 0
        if valid_lengths_mask.any():
            # gather_indices باید از original_sequence_lens برای توالی‌های معتبر باشد
            # و باید کوچکتر از طول پد شده transformer_output.size(1) باشد
            actual_lengths_for_ht = original_sequence_lens[valid_lengths_mask]
            # اطمینان از اینکه اندیس از طول پد شده تجاوز نمی‌کند
            gather_indices = (actual_lengths_for_ht - 1).clamp(min=0, max=transformer_output.size(1)-1)

            batch_indices_ht = torch.arange(batch_size, device=device)[valid_lengths_mask]

            if transformer_output.size(1) > 0 and gather_indices.numel() > 0:
                 # بررسی اینکه آیا gather_indices برای همه batch_indices_ht معتبر است
                 # این بخش ممکن است نیاز به دقت بیشتری در اندیس‌گذاری داشته باشد
                 # اگر برخی توالی‌ها کاملاً خالی باشند (original_sequence_lens=0)
                 # ht_transformer[batch_indices_ht] = transformer_output[batch_indices_ht, gather_indices]
                 # یک راه امن‌تر:
                 for idx, b_idx in enumerate(batch_indices_ht):
                     g_idx = gather_indices[idx]
                     ht_transformer[b_idx] = transformer_output[b_idx, g_idx]


        # محاسبه a (نمایش سراسری جلسه از خروجی Transformer)
        q1 = self.linear_one(ht_transformer).view(batch_size, 1, self.hidden_size)
        q2 = self.linear_two(transformer_output)

        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2))
        # ~attention_mask_transformer چون masked_fill جایی که True است را پر می‌کند
        alpha_logits_masked = alpha_logits.masked_fill(~attention_mask_transformer.unsqueeze(-1), -float('inf'))
        alpha = F.softmax(alpha_logits_masked, dim=1)

        a_transformer = torch.sum(alpha * transformer_output * mask_float_transformer.unsqueeze(-1), dim=1)

        # --- انتخاب نمایش نهایی برای محاسبه امتیاز ---
        if self.nonhybrid:
            final_session_repr = self.linear_transform(torch.cat([a_transformer, ht_transformer], 1))
        else:
            # پیاده‌سازی Target Attention از کد اصلی شما در اینجا لازم است
            # ورودی‌ها: a_transformer, transformer_output (به جای hidden_transformer_output قبلی),
            # و attention_mask_transformer (به جای mask قبلی)
            # این بخش نیاز به بازنویسی دقیق دارد.
            #
            # برای این مثال، از یک ترکیب ساده استفاده می‌کنیم (مانند حالت nonhybrid)
            # **شما باید این بخش را با منطق Target Attention خودتان جایگزین کنید.**
            final_session_repr = self.linear_transform(torch.cat([a_transformer, ht_transformer], 1))


        candidate_item_embeddings = self.embedding.weight[1:]
        scores = torch.matmul(final_session_repr, candidate_item_embeddings.t())
        return scores

    # def calculate_infonce_loss(self, z_i, z_j, temperature): # یک نمونه ساده
    #     batch_size = z_i.size(0)
    #     z_i_norm = F.normalize(z_i, p=2, dim=1)
    #     z_j_norm = F.normalize(z_j, p=2, dim=1)
    #     representations = torch.cat([z_i_norm, z_j_norm], dim=0)
    #     similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
    #     sim_ij = torch.diag(similarity_matrix, batch_size)
    #     sim_ji = torch.diag(similarity_matrix, -batch_size)
    #     positives = torch.cat([sim_ij, sim_ji], dim=0)
        
    #     nominator = torch.exp(positives / temperature)
        
    #     # ایجاد ماسک برای حذف نمونه‌های مثبت از مخرج (خود نمونه و زوج آگمنت شده‌اش)
    #     diag_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    #     # این بخش نیاز به دقت بیشتری برای انتخاب درست نمونه‌های منفی دارد.
    #     # این یک پیاده‌سازی بسیار ساده است.
    #     denominator = torch.sum(torch.exp(similarity_matrix[diag_mask].view(2*batch_size, -1) / temperature), dim=1)
        
    #     loss = -torch.log(nominator / denominator)
    #     return loss.mean()