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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        
        # Initialize parameters with Xavier uniform
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
        src2 = self.norm1(src)
        sa_output, _ = self.self_attn(src2, src2, src2,
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
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
        self.norm = norm

    def forward(self, src, candidate_embeddings_global, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, candidate_embeddings_global, 
                        src_mask=mask, 
                        src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.nonhybrid = opt.nonhybrid
        self.ssl_weight = opt.ssl_weight
        self.ssl_temp = opt.ssl_temp
        self.ssl_dropout_rate = opt.ssl_dropout_rate
        
        # Initialize embeddings
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)
        
        # Initialize modules
        self.gnn = GNN(self.hidden_size, step=opt.step)
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
        
        # Prediction layers
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Optimization
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
        
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def compute_scores(self, hidden_transformer_output, mask):
        mask = mask.float()
        seq_len = hidden_transformer_output.size(1)
        
        # Get last valid item
        ht = hidden_transformer_output[torch.arange(mask.size(0)), 
                                      torch.clamp(mask.sum(1) - 1, min=0).long()]
        
        # Attention mechanism
        q1 = self.linear_one(ht).unsqueeze(1)
        q2 = self.linear_two(hidden_transformer_output)
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1)
        
        # Masked softmax
        alpha_logits_masked = alpha_logits.masked_fill(mask == 0, torch.finfo(alpha_logits.dtype).min)
        alpha = torch.softmax(alpha_logits_masked, dim=1)
        
        # Weighted sum
        a = (alpha.unsqueeze(-1) * hidden_transformer_output * mask.unsqueeze(-1)).sum(1)
        
        # Get candidate embeddings
        candidate_embeds = self.embedding.weight[1:]  # exclude padding
        
        if self.nonhybrid:
            combined = self.linear_transform(torch.cat([a, ht], 1))
            scores = torch.matmul(combined, candidate_embeds.t())
        else:
            qt = self.linear_t(hidden_transformer_output)
            beta_logits = torch.matmul(candidate_embeds, qt.transpose(1, 2))
            beta_logits_masked = beta_logits.masked_fill(mask.unsqueeze(1) == 0, torch.finfo(beta_logits.dtype).min)
            beta = torch.softmax(beta_logits_masked, dim=-1)
            
            target_ctx = torch.matmul(beta, qt * mask.unsqueeze(-1))
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
        
        return (loss_12.mean() + loss_21.mean()) / 2.0

def forward(model, i, data, is_train=True):
    alias_inputs_np, A_np, items_np, mask_np, targets_np = data.get_slice(i)
    device = next(model.parameters()).device
    
    # Convert to tensors and move to device
    alias_inputs = torch.from_numpy(alias_inputs_np).long().to(device)
    A = torch.from_numpy(A_np).float().to(device)
    items = torch.from_numpy(items_np).long().to(device)
    mask = torch.from_numpy(mask_np).float().to(device)
    targets = torch.from_numpy(targets_np).long().to(device)
    
    # Forward pass through GNN
    hidden = model.embedding(items)
    hidden = model.gnn(A, hidden)
    
    # Prepare sequence input for transformer
    seq_hidden = torch.gather(hidden, 1, 
                             alias_inputs.unsqueeze(-1).expand(-1, -1, model.hidden_size))
    
    # Add positional encoding
    seq_hidden = model.pos_encoder(seq_hidden)
    
    # Transformer encoder
    src_key_padding_mask = (mask == 0)
    output = model.transformer_encoder(
        src=seq_hidden,
        candidate_embeddings_global=model.embedding.weight[1:],
        src_key_padding_mask=src_key_padding_mask
    )
    
    # Compute scores
    scores = model.compute_scores(output, mask)
    
    # SSL loss (optional)
    ssl_loss = torch.tensor(0.0, device=device)
    if is_train and model.ssl_weight > 0:
        try:
            # Get last valid item embeddings
            last_idx = torch.clamp(mask.sum(1) - 1, min=0).long()
            ssl_base_emb = seq_hidden[torch.arange(mask.size(0)), last_idx]
            
            # Create two views with dropout
            ssl_emb1 = F.dropout(ssl_base_emb, p=model.ssl_dropout_rate, training=True)
            ssl_emb2 = F.dropout(ssl_base_emb, p=model.ssl_dropout_rate, training=True)
            
            ssl_loss = model.calculate_ssl_loss(ssl_emb1, ssl_emb2, model.ssl_temp)
        except Exception as e:
            print(f"SSL calculation error: {e}")
            ssl_loss = torch.tensor(0.0, device=device)
    
    return targets, scores, ssl_loss

def train_test(model, train_data, test_data, opt):
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_ir = now_utc.astimezone(IR_TIMEZONE)
    print(f'start training: {now_ir.strftime("%Y-%m-%d %H:%M:%S")}')
    
    model.train()
    total_loss = 0.0
    total_rec_loss = 0.0
    total_ssl_loss = 0.0
    
    slices = train_data.generate_batch(opt.batchSize)
    for step, i in enumerate(slices):
        model.optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=torch.cuda.is_available()):
            targets, scores, ssl_loss = forward(model, i, train_data, is_train=True)
            
            # Calculate recommendation loss
            valid_targets = (targets > 0) & (targets <= model.n_node)
            rec_loss = torch.tensor(0.0, device=scores.device)
            
            if valid_targets.any():
                target_values = (targets[valid_targets] - 1).clamp(0, scores.size(1) - 1)
                rec_loss = model.loss_function(scores[valid_targets], target_values)
            
            total_loss_val = rec_loss + model.ssl_weight * ssl_loss
        
        # Backward pass
        if torch.cuda.is_available():
            scaler.scale(total_loss_val).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            total_loss_val.backward()
            model.optimizer.step()
        
        # Update statistics
        total_loss += total_loss_val.item()
        total_rec_loss += rec_loss.item()
        total_ssl_loss += ssl_loss.item()
        
        if (step + 1) % max(1, int(len(slices) / 5)) == 0:
            avg_loss = total_loss / (step + 1)
            avg_rec_loss = total_rec_loss / (step + 1)
            avg_ssl_loss = total_ssl_loss / (step + 1)
            print(f'[{step + 1}/{len(slices)}] Tot Loss: {avg_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}, SSL Loss: {avg_ssl_loss:.4f}')
    
    model.scheduler.step()
    
    # Evaluation
    model.eval()
    hit, mrr, precision = [], [], []
    k_metric = 20
    
    test_slices = test_data.generate_batch(opt.batchSize)
    with torch.no_grad():
        for i in test_slices:
            targets, scores, _ = forward(model, i, test_data, is_train=False)
            _, indices = scores.topk(k_metric, dim=1)
            
            indices = indices.cpu().numpy()
            targets = targets.cpu().numpy()
            
            for pred, target in zip(indices, targets):
                if target > 0:
                    target_idx = target - 1
                    hit.append(np.isin(target_idx, pred))
                    if hit[-1]:
                        rank = np.where(pred == target_idx)[0][0] + 1
                        mrr.append(1.0 / rank)
                        precision.append(1.0 / k_metric)
                    else:
                        mrr.append(0.0)
                        precision.append(0.0)
    
    hit_rate = np.mean(hit) * 100 if hit else 0.0
    mrr_score = np.mean(mrr) * 100 if mrr else 0.0
    precision_score = np.mean(precision) * 100 if precision else 0.0
    
    return hit_rate, mrr_score, precision_score