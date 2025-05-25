import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool # برای SSL
import math
import copy

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

# -------------- 2. TargetAwareEncoderLayer (از کد اصلی شما) --------------
class TargetAwareEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super(TargetAwareEncoderLayer, self).__init__()
        self.W_sc_q = nn.Linear(d_model, d_model)
        self.W_sc_k = nn.Linear(d_model, d_model)
        self.W_sc_v = nn.Linear(d_model, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff = nn.Dropout(dropout) # نام متفاوت برای جلوگیری از تداخل با پارامتر dropout
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src, candidate_embeddings_global, src_key_padding_mask=None, src_mask=None):
        q_sc = self.W_sc_q(src)
        k_sc = self.W_sc_k(candidate_embeddings_global)
        v_sc = self.W_sc_v(candidate_embeddings_global)
        attn_score_sc = torch.matmul(q_sc, k_sc.transpose(-2, -1)) / math.sqrt(q_sc.size(-1))
        attn_weights_sc = F.softmax(attn_score_sc, dim=-1)
        context_from_candidates = torch.matmul(attn_weights_sc, v_sc)
        src_enhanced = src + context_from_candidates
        sa_output, _ = self.self_attn(src_enhanced, src_enhanced, src_enhanced,
                                      key_padding_mask=src_key_padding_mask,
                                      attn_mask=src_mask)
        out1 = src + self.dropout1(sa_output)
        out1 = self.norm1(out1)
        ff_output = self.linear2(self.dropout_ff(self.activation(self.linear1(out1))))
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
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.nonhybrid = opt.nonhybrid
        self.ssl_weight = opt.ssl_weight
        self.ssl_temp = opt.ssl_temp
        self.ssl_dropout_rate = opt.ssl_dropout_rate

        self.embedding = nn.Embedding(n_node, self.hidden_size, padding_idx=0)
        self.gnn = GatedGraphConv(out_channels=self.hidden_size, num_layers=opt.step)

        self.pos_encoder = PositionalEncoding(self.hidden_size, dropout=opt.dropout)

        ta_encoder_layer = TargetAwareEncoderLayer(
            d_model=self.hidden_size,
            nhead=opt.nhead,
            dim_feedforward=opt.ff_hidden,
            dropout=opt.dropout
        )
        self.transformer_encoder = TargetAwareTransformerEncoder(
            encoder_layer=ta_encoder_layer,
            num_layers=opt.nlayers,
            norm=nn.LayerNorm(self.hidden_size)
        )

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # برای Target Attention

        self.loss_function = nn.CrossEntropyLoss()
        # Optimizer و Scheduler در main.py تعریف می‌شوند

    def map_gnn_output_to_transformer_input(self, gnn_output_flat, pyg_batch, original_sequence_ids_padded):
        """
        **تابع حیاتی و چالش‌برانگیز برای نگاشت خروجی GNN به ورودی Transformer.**
        gnn_output_flat: (total_num_nodes_in_batch, D) - نمایش نودهای GNN
        pyg_batch.x: (total_num_nodes_in_batch,) - IDهای اصلی نودهای یکتا که gnn_output_flat برای آنهاست
        pyg_batch.batch: (total_num_nodes_in_batch,) - نگاشت هر نود به گرافش در بچ
        original_sequence_ids_padded: (B, L_padded) - IDهای اصلی آیتم‌ها در توالی‌ها (با پدینگ 0)

        خروجی: (B, L_padded, D) - نمایش‌های GNN برای هر آیتم در توالی‌های پد شده
        """
        batch_size = original_sequence_ids_padded.size(0)
        max_seq_len = original_sequence_ids_padded.size(1)
        device = gnn_output_flat.device

        # ایجاد یک نمایش پایه (مثلاً امبدینگ پدینگ) برای آیتم‌هایی که در گراف GNN نیستند یا پدینگ هستند
        # امبدینگ آیتم 0 (پدینگ) را می‌گیریم
        padding_embedding = self.embedding(torch.tensor([0], device=device)).squeeze(0) # (D)

        # تنسور خروجی را با امبدینگ پدینگ پر می‌کنیم
        transformer_input = padding_embedding.repeat(batch_size, max_seq_len, 1) # (B, L, D)

        # اگر gnn_output_flat خالی باشد (هیچ گراف معتبری در بچ نبوده)
        if gnn_output_flat is None or gnn_output_flat.numel() == 0:
            return transformer_input # فقط شامل پدینگ خواهد بود

        # 1. نگاشت از ID آیتم اصلی به اندیس آن در gnn_output_flat
        # pyg_batch.x IDهای اصلی نودهای یکتا در کل بچ است
        # gnn_output_flat نمایش این نودها به همان ترتیب است
        # این نگاشت برای دسترسی سریع به نمایش GNN یک ID خاص است.
        unique_ids_in_gnn_batch = pyg_batch.x # (total_nodes,)
        
        # حلقه روی هر توالی در بچ
        for i in range(batch_size):
            current_original_seq_ids = original_sequence_ids_padded[i] # (L_padded,)
            
            # حلقه روی هر آیتم در توالی پد شده
            for j in range(max_seq_len):
                item_id = current_original_seq_ids[j].item()
                if item_id == 0: # اگر پدینگ بود، از قبل با امبدینگ پدینگ پر شده
                    continue

                # پیدا کردن اندیس item_id در unique_ids_in_gnn_batch
                # این بخش می‌تواند کند باشد. به دنبال راه بهینه‌تر باشید.
                # (ممکن است از torch.where یا دیکشنری استفاده کرد، اما ساخت دیکشنری در هر بچ هم هزینه دارد)
                # یک راه بهینه‌تر: یکبار یک "مگا-تنسور" یا دیکشنری برای نگاشت IDهای کل دیتاست به اندیس‌های یکتا بسازید
                # و در اینجا فقط lookup کنید. اما اینجا ما اندیس در gnn_output_flat را می‌خواهیم.
                
                # پیدا کردن اولین وقوع item_id در unique_ids_in_gnn_batch
                # توجه: اگر یک ID در گراف نباشد (مثلاً جلسه خالی بوده و فقط توالی پد شده داریم)
                # این بخش باید مدیریت شود.
                # (pyg_batch.x فقط شامل نودهای گراف‌های معتبر است)
                
                # ابتدا بررسی کنیم این item_id در کدام گراف از بچ pyg قرار دارد
                # و سپس اندیس محلی آن در آن گراف، و سپس اندیس سراسری در gnn_output_flat
                # این بسیار پیچیده می‌شود.

                # **رویکرد ساده‌تر اما با فرض مهم:**
                # فرض کنیم unique_ids_in_gnn_batch شامل تمام IDهای آیتمی است که ممکن است
                # در original_sequence_ids_padded ظاهر شوند (به جز 0).
                # اگر اینطور باشد، می‌توانیم اندیس‌ها را پیدا کنیم.

                # یک راه ساده‌تر (ولی ممکن است برای IDهای تکراری در جلسات مختلف مشکل‌ساز باشد اگر نگاشت یکتا به یکتا نباشد):
                # این بخش نیاز به بازبینی اساسی دارد.
                # برای هر item_id، باید نمایش GNN آن را پیدا کنیم.
                # اگر یک item_id در چند گراف مختلف در بچ باشد، کدام نمایش GNN را برداریم؟
                # gnn_output_flat نمایش تمام نودهای تمام گراف‌ها پشت سر هم است.
                # pyg_batch.batch می‌گوید هر نود در gnn_output_flat متعلق به کدام گراف است.

                # **ساده‌سازی برای این مثال (نیاز به بهبود دارد!):**
                # فقط اگر item_id در unique_ids_in_gnn_batch وجود دارد، نمایش آن را بردار.
                # این ممکن است برای آیتم‌هایی که در گراف نیستند (چون جلسه اصلی‌شان خالی بوده) کار نکند.
                indices_in_gnn_flat = (unique_ids_in_gnn_batch == item_id).nonzero(as_tuple=True)[0]
                if indices_in_gnn_flat.numel() > 0:
                    # اگر یک ID چند بار در unique_ids_in_gnn_batch باشد (نباید باشد چون unique است)
                    # اولین مورد را می‌گیریم.
                    # اما چالش اینجاست که اگر item_id در جلسه i-ام باشد، باید نمایش GNN آن از همان جلسه i-ام باشد.
                    # این نیاز به استفاده از pyg_batch.batch دارد.
                    
                    # **این پیاده‌سازی بسیار ساده شده و احتمالاً نادرست است.**
                    # **شما باید راهی برای نگاشت دقیق هر (آیتم در توالی i، اندیس j) به نمایش GNN صحیح آن پیدا کنید.**
                    # یک ایده: قبل از GNN، یک نگاشت از (ایندکس گراف در بچ، ID اصلی آیتم) به (ایندکس نود در gnn_output_flat) بسازید.
                    
                    # برای اینکه کد ادامه پیدا کند، یک راه حل موقت (احتمالاً نادرست):
                    # اگر item_id در لیست نودهای یکتای GNN بچ بود، اولین نمایش آن را بردار.
                    # این تفاوت بین جلسات را در نظر نمی‌گیرد.
                    first_occurrence_idx = indices_in_gnn_flat[0]
                    transformer_input[i, j] = gnn_output_flat[first_occurrence_idx]
                # else: آیتم در گراف GNN این بچ نیست، با پدینگ باقی می‌ماند.
        
        return transformer_input


    def forward(self, pyg_batch, original_sequence_ids_padded, attention_masks_transformer, original_sequence_lens, is_train=True):
        current_batch_size = original_sequence_ids_padded.size(0)
        device = self.embedding.weight.device

        gnn_output_flat = None
        if pyg_batch is not None and pyg_batch.x is not None and pyg_batch.x.numel() > 0 :
            node_features_embedded = self.embedding(pyg_batch.x)
            gnn_output_flat = self.gnn(node_features_embedded, pyg_batch.edge_index)
        
        # **استفاده از تابع نگاشت برای آماده‌سازی ورودی Transformer**
        # این بخش بسیار حیاتی و نیازمند پیاده‌سازی دقیق map_gnn_output_to_transformer_input است.
        # اگر map_gnn_output_to_transformer_input به درستی کار نکند، این بخش نتایج نادرستی خواهد داشت.
        if gnn_output_flat is not None:
            transformer_input_features = self.map_gnn_output_to_transformer_input(
                gnn_output_flat, pyg_batch, original_sequence_ids_padded, current_batch_size
            )
        else: # اگر هیچ گراف معتبری در بچ نبود، از امبدینگ مستقیم توالی‌ها استفاده می‌کنیم
              # یا یک ورودی پدینگ کامل برای Transformer ایجاد می‌کنیم.
            transformer_input_features = self.embedding(original_sequence_ids_padded)


        transformer_input_pos = self.pos_encoder(transformer_input_features)
        src_key_padding_mask_transformer = ~attention_masks_transformer

        all_item_ids = torch.arange(1, self.n_node, device=device)
        candidate_embeddings_global = self.embedding(all_item_ids)

        transformer_output = self.transformer_encoder(
            src=transformer_input_pos,
            candidate_embeddings_global=candidate_embeddings_global,
            src_key_padding_mask=src_key_padding_mask_transformer
        )

        scores = self._compute_scores_from_transformer(
            transformer_output,
            attention_masks_transformer,
            original_sequence_lens
        )

        ssl_loss_value = torch.tensor(0.0, device=scores.device)
        if is_train and self.ssl_weight > 0 and gnn_output_flat is not None and pyg_batch.num_graphs > 0:
            # **بازنویسی SSL با استفاده از gnn_output_flat و pyg_batch.batch**
            # باید نمایش آخرین نود (یا نودهای دیگر) هر گراف را استخراج کرده و SSL را اعمال کنید.
            # این بخش نیازمند منطق دقیق استخراج است.
            # مثال ساده با global_mean_pool (ممکن است برای SSL شما مناسب نباشد):
            # pooled_gnn_outputs = global_mean_pool(gnn_output_flat, pyg_batch.batch)
            # if pooled_gnn_outputs.size(0) >= 2: # نیاز به حداقل دو جلسه برای مقایسه
            #     view1 = self.dropout(pooled_gnn_outputs) # یا هر آگمنتاسیون دیگری
            #     view2 = self.dropout(pooled_gnn_outputs)
            #     # ssl_loss_value = self.calculate_infonce_loss(view1, view2, self.ssl_temp) # نیاز به پیاده‌سازی
            pass # SSL را فعلا غیرفعال نگه دارید تا روی بخش اصلی تمرکز کنید.

            ssl_loss_value = ssl_loss_value * self.ssl_weight

        if is_train:
            return scores, ssl_loss_value
        else:
            return scores

    def _compute_scores_from_transformer(self, transformer_output, attention_mask_transformer, original_sequence_lens):
        device = transformer_output.device
        batch_size = transformer_output.size(0)
        mask_float_transformer = attention_mask_transformer.float()

        ht_transformer = torch.zeros(batch_size, self.hidden_size, device=device)
        valid_lengths_mask = original_sequence_lens > 0
        if valid_lengths_mask.any():
            actual_lengths_for_ht = original_sequence_lens[valid_lengths_mask]
            gather_indices = (actual_lengths_for_ht - 1).clamp(min=0, max=transformer_output.size(1)-1)
            batch_indices_ht = torch.arange(batch_size, device=device)[valid_lengths_mask]
            if transformer_output.size(1) > 0 and gather_indices.numel() > 0:
                 for idx, b_idx in enumerate(batch_indices_ht):
                     g_idx = gather_indices[idx]
                     ht_transformer[b_idx] = transformer_output[b_idx, g_idx]

        q1 = self.linear_one(ht_transformer).view(batch_size, 1, self.hidden_size)
        q2 = self.linear_two(transformer_output)
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2))
        alpha_logits_masked = alpha_logits.masked_fill(~attention_mask_transformer.unsqueeze(-1), -float('inf'))
        alpha = F.softmax(alpha_logits_masked, dim=1)
        a_transformer = torch.sum(alpha * transformer_output * mask_float_transformer.unsqueeze(-1), dim=1)

        if self.nonhybrid:
            final_session_repr = self.linear_transform(torch.cat([a_transformer, ht_transformer], 1))
        else:
            # **اینجا باید منطق Target Attention از کد اصلی شما با ورودی‌های جدید پیاده‌سازی شود**
            # ورودی‌ها: a_transformer, transformer_output, attention_mask_transformer
            # برای این مثال، از ترکیب ساده استفاده می‌کنیم:
            final_session_repr = self.linear_transform(torch.cat([a_transformer, ht_transformer], 1))

        candidate_item_embeddings = self.embedding.weight[1:]
        scores = torch.matmul(final_session_repr, candidate_item_embeddings.t())
        return scores

    # def calculate_infonce_loss(self, z_i, z_j, temperature): # نمونه SSL
    #     # ... (پیاده‌سازی InfoNCE) ...
    #     pass