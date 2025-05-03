import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
# اضافه کردن ایمپورت های لازم برای Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# -------------- 1. اضافه کردن کلاس PositionalEncoding --------------
class PositionalEncoding(Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe) # bufer برای اینکه جزو پارامترهای مدل نباشد

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
               shape: (sequence length, batch size, embed dim)
        Shape:
            output: (sequence length, batch size, embed dim)
        """
        # x.size(0) is sequence length
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
# -------------------------------------------------------------------


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
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
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
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)

        # -------------- 2. اضافه کردن لایه Transformer --------------
        # پارامترهای Transformer (می توانید تنظیم کنید)
        nhead = opt.nhead if hasattr(opt, 'nhead') else 2 # تعداد head های attention
        num_encoder_layers = opt.nlayers if hasattr(opt, 'nlayers') else 2 # تعداد لایه های encoder
        dim_feedforward = opt.ff_hidden if hasattr(opt, 'ff_hidden') else 256 # ابعاد لایه feedforward
        dropout = opt.dropout if hasattr(opt, 'dropout') else 0.1 # مقدار dropout

        self.pos_encoder = PositionalEncoding(self.hidden_size, dropout)
        encoder_layers = TransformerEncoderLayer(self.hidden_size, nhead, dim_feedforward, dropout, batch_first=True) # batch_first=True مهم است
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        # ----------------------------------------------------------

        # لایه های قبلی برای محاسبه score (ممکن است برخی نیاز به تغییر داشته باشند)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        # توجه: ممکن است linear_transform دیگر لازم نباشد یا نیاز به تغییر ورودی داشته باشد
        # اگر nonhybrid نباشد، همچنان last item embedding را اضافه می کنیم
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            # از مقدار دهی اولیه Xavier/Glorot برای Transformer استفاده می کنیم اگر وجود داشت
            if weight.dim() > 1 and 'transformer' in self._modules and hasattr(self._modules['transformer_encoder'], 'parameters'):
                 if weight in self._modules['transformer_encoder'].parameters():
                      nn.init.xavier_uniform_(weight)
                      continue # برو سراغ وزن بعدی
            # مقداردهی اولیه قبلی برای بقیه پارامترها
            weight.data.uniform_(-stdv, stdv)

    # compute_scores حالا ورودی اش خروجی Transformer خواهد بود
    def compute_scores(self, hidden_transformer_output, mask):
        # hidden_transformer_output shape: (batch_size, seq_length, hidden_size)
        ht = hidden_transformer_output[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size (last item embed from transformer)

        # --- محاسبه session representation با استفاده از خروجی Transformer ---
        # روش 1: استفاده از مکانیزم توجه اصلی روی خروجی Transformer
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden_transformer_output)
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) # (b,s,1)
        # alpha = torch.sigmoid(alpha)
        alpha = F.softmax(alpha, 1) # B,S,1
        a = torch.sum(alpha * hidden_transformer_output * mask.view(mask.shape[0], -1, 1).float(), 1) # (b,d) - Global preference

        # روش 2: استفاده از میانگین خروجی های Transformer (ساده تر)
        # seq_len_per_batch = torch.sum(mask, 1).unsqueeze(1).float() # (b, 1)
        # a = torch.sum(hidden_transformer_output * mask.view(mask.shape[0], -1, 1).float(), 1) / seq_len_per_batch # (b,d) - Global preference (mean pooling)
        # --------------------------------------------------------------------

        # ترکیب با last item embedding (مانند کد اصلی)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1)) # (b,d) - Final session preference before target attention

        # --- Target Attention روی خروجی Transformer ---
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        # mask  # batch_size x seq_length
        # hidden_transformer_output را mask می کنیم
        hidden_masked = hidden_transformer_output * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size
        qt = self.linear_t(hidden_masked)  # batch_size x seq_length x latent_size
        beta = F.softmax(b @ qt.transpose(1, 2), -1)  # batch_size x n_nodes x seq_length
        target = beta @ hidden_masked  # batch_size x n_nodes x latent_size
        # ----------------------------------------------

        # ترکیب session preference با target-specific representation
        final_representation = a.view(ht.shape[0], 1, ht.shape[1]) + target  # b,n,d

        # محاسبه امتیاز نهایی
        scores = torch.sum(final_representation * b, -1)  # b,n
        # scores = torch.matmul(final_representation, b.transpose(1, 0)) # جایگزین برای محاسبه score
        return scores

    # تابع forward اصلی مدل بدون تغییر باقی می ماند، فقط GNN را اجرا می کند
    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


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


# -------------- 3. تغییر در تابع forward (خارج از کلاس) --------------
def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    hidden = model.embedding(items)
    hidden = model.gnn(A, hidden)

    # برداری‌سازی و اعمال Positional Encoding
    seq_hidden = hidden[torch.arange(len(alias_inputs)).unsqueeze(1), alias_inputs]  # (batch_size, seq_len, hidden_size)
    seq_hidden_pos = model.pos_encoder(seq_hidden.transpose(0, 1)).transpose(0, 1)  # (batch_size, seq_len, hidden_size)
    
    # ساخت ماسک با شکل صحیح (batch_size, seq_len)
    src_key_padding_mask = (mask == 0)
    
    # اجرای Transformer
    hidden_transformer_output = model.transformer_encoder(
        src=seq_hidden_pos,
        src_key_padding_mask=src_key_padding_mask
    )
    
    return targets, model.compute_scores(hidden_transformer_output, mask)
# ---------------------------------------------------------------------


from torch.cuda.amp import autocast, GradScaler

 

def train_test(model, train_data, test_data):
    scaler = GradScaler()  # تغییر به سینتکس جدید
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        with autocast(  dtype=torch.float16):  # تغییر به سینتکس جدید
            targets, scores = forward(model, i, train_data)
            targets = trans_to_cuda(torch.Tensor(targets).long())
            loss = model.loss_function(scores, targets - 1)
        scaler.scale(loss).backward()
        scaler.step(model.optimizer)
        scaler.update()
        total_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data) # از تابع forward اصلاح شده استفاده می شود
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        # محاسبه Hit Rate و MRR مثل قبل
        targets_cpu = trans_to_cpu(targets).detach().numpy() # اضافه شد برای دسترسی به mask
        mask_cpu = trans_to_cpu(test_data.mask[i]).detach().numpy() # اضافه شد

        for score, target, mask_row in zip(sub_scores, targets_cpu, mask_cpu): # mask اضافه شد
             # اطمینان از اینکه target معتبر است (در برخی موارد ممکن است 0 باشد اگر ورودی خالی باشد)
             if target > 0:
                  hit.append(np.isin(target - 1, score))
                  if len(np.where(score == target - 1)[0]) == 0:
                      mrr.append(0)
                  else:
                      mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100 if hit else 0
    mrr = np.mean(mrr) * 100 if mrr else 0
    return hit, mrr
