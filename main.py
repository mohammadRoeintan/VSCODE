import torch
from torch.optim import Adam
from model_pyg_advanced import SessionGraph # نام فایل مدل جدید
from utils_pyg_advanced import generate_dataloader_pyg_advanced # نام فایل utils جدید
import numpy as np
import os
import pickle

# --- پارامترها (باید از argparse مثل کد اصلی شما بیایند) ---
class OptMock: # یک کلاس ساده برای شبیه‌سازی opt از argparse
    def __init__(self):
        self.dataset = 'sample_pyg_adv'
        self.hiddenSize = 100
        self.batchSize = 64 # کوچکتر برای تست اولیه
        self.epoch = 5
        self.lr = 0.001
        self.step = 2 # تعداد لایه‌های GNN (num_gnn_steps)
        self.nhead = 2 # برای TargetAware (nhead_transformer)
        self.nlayers = 1 # برای TargetAware (nlayers_transformer)
        self.dropout = 0.2 # برای PositionalEncoding و TargetAware (dropout_transformer)
        self.ff_hidden = 256 # برای TargetAware (dim_feedforward)
        self.ssl_weight = 0.0 # فعلا SSL غیرفعال
        self.ssl_temp = 0.5
        self.ssl_dropout_rate = 0.2
        self.nonhybrid = False
        self.n_node = 310 # مقدار پیش‌فرض, باید با داده شما تنظیم شود

opt = OptMock()
# -------------------------------------------------------------

# --- بارگذاری داده‌ها ---
data_dir = "./datasets" # مسیر دیتاست خود را تنظیم کنید
train_sessions, train_targets = [], []
test_sessions, test_targets = [], []
try:
    with open(os.path.join(data_dir, opt.dataset, "train.txt"), 'rb') as f:
        train_sessions, train_targets = pickle.load(f)
    with open(os.path.join(data_dir, opt.dataset, "test.txt"), 'rb') as f:
        test_sessions, test_targets = pickle.load(f)
except FileNotFoundError:
    print(f"Warning: Data files not found in {os.path.join(data_dir, opt.dataset)}")
    print("Creating dummy data for 'sample_pyg_adv'...")
    train_sessions = [[1,2,3,4], [2,3,4,2,5], [5,1,6], [6,7,8,9,6,10], [11,12],[1],[1,2]]
    train_targets = [5,1,7,11,1,2,3]
    test_sessions = [[1,2,3], [3,4,5,6],[7]]
    test_targets = [4,2,8]
    all_items = set(sum(train_sessions, []) + sum(test_sessions, []) + train_targets + test_targets)
    if all_items: opt.n_node = max(all_items) + 1
    else: opt.n_node = 20
    print(f"Using dummy data with n_node = {opt.n_node}")
# -------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if opt.n_node <=1 : opt.n_node = 20 # حداقل مقدار برای n_node

model = SessionGraph(opt, n_node=opt.n_node).to(device) # پاس دادن opt به مدل
optimizer = Adam(model.parameters(), lr=opt.lr)

train_loader = generate_dataloader_pyg_advanced(train_sessions, train_targets, batch_size=opt.batchSize, shuffle=True)
test_loader = generate_dataloader_pyg_advanced(test_sessions, test_targets, batch_size=opt.batchSize, shuffle=False)

checkpoint_dir = f"./checkpoints_{opt.dataset}"
os.makedirs(checkpoint_dir, exist_ok=True)
best_recall_at_20 = 0.0

print(f"Starting training for {opt.epoch} epochs...")
for epoch in range(opt.epoch):
    model.train()
    total_loss_epoch = 0
    total_rec_loss_epoch = 0
    total_ssl_loss_epoch = 0
    num_batches_train = 0

    for batch_data in train_loader:
        # collate_fn ممکن است None برای pyg_batch برگرداند اگر تمام جلسات خالی بودند
        if batch_data[0] is None: # batch_data[0] همان pyg_batch است
            # print("Skipping a batch with no valid graphs.")
            continue
        
        pyg_batch, original_sequences_padded, attention_masks, targets, original_sequence_lens = batch_data

        # اگر به هر دلیلی، پس از collate، بچ خالی شد (مثلاً فقط جلسات خالی داشتیم)
        if pyg_batch is not None and (pyg_batch.x is None or pyg_batch.x.numel() == 0) and original_sequences_padded.numel() == 0:
            # print("Skipping completely empty batch after collate.")
            continue


        # انتقال همه چیز به دستگاه
        if pyg_batch is not None: pyg_batch = pyg_batch.to(device)
        original_sequences_padded = original_sequences_padded.to(device)
        attention_masks = attention_masks.to(device)
        targets = targets.to(device).squeeze() # (B)
        original_sequence_lens = original_sequence_lens.to(device)


        optimizer.zero_grad(set_to_none=True)
        scores, ssl_loss = model(pyg_batch, original_sequences_padded, attention_masks, original_sequence_lens, is_train=True)

        # مدیریت بچ خالی در خروجی مدل
        if scores is None or scores.numel() == 0:
            # print("Skipping batch due to empty scores from model.")
            continue


        valid_targets_mask = (targets > 0) # فقط اهداف معتبر
        num_valid_targets = valid_targets_mask.sum()

        rec_loss_batch = torch.tensor(0.0, device=device)
        if num_valid_targets > 0:
            # scores باید (num_valid_targets_in_batch, n_node-1) باشد اگر بچ فیلتر شده
            # یا (B, n_node-1) و targets (B) باشد
            # در اینجا scores برای تمام B آیتم بچ است (حتی آنها که جلسه GNN خالی داشتند)
            # اما targets هم برای تمام B آیتم بچ است.
            # پس valid_targets_mask را روی هر دو اعمال می‌کنیم.
            
            # اطمینان از اینکه scores و targets[valid_targets_mask] ابعاد سازگار دارند
            # scores باید برای تمام نمونه‌های بچ باشد، targets هم
            # scores (B, n_node-1)
            # targets (B) شامل IDهای از 1 تا n_node
            # target_for_loss (B_valid) شامل IDهای از 0 تا n_node-2
            
            # فقط برای نمونه‌هایی که هدف معتبر دارند loss را محاسبه می‌کنیم
            current_scores = scores[valid_targets_mask]
            current_targets_for_loss = (targets[valid_targets_mask] - 1).clamp(0, opt.n_node - 2)

            if current_scores.shape[0] > 0 and current_scores.shape[0] == current_targets_for_loss.shape[0]:
                 rec_loss_batch = model.loss_function(current_scores, current_targets_for_loss)
            # else:
                 # print(f"Skipping loss for batch due to shape mismatch or no valid targets after filtering. Scores shape: {current_scores.shape}, Targets shape: {current_targets_for_loss.shape}")


        loss_batch = rec_loss_batch + ssl_loss # ssl_loss از مدل با وزن می‌آید

        if torch.isnan(loss_batch) or torch.isinf(loss_batch):
            # print(f"NaN or Inf loss detected: {loss_batch.item()}. Skipping backward.")
            continue

        loss_batch.backward()
        optimizer.step()

        total_loss_epoch += loss_batch.item()
        total_rec_loss_epoch += rec_loss_batch.item() # rec_loss_batch می‌تواند صفر باشد
        total_ssl_loss_epoch += ssl_loss.item()
        num_batches_train += 1

    avg_total_loss = total_loss_epoch / num_batches_train if num_batches_train > 0 else 0
    avg_rec_loss = total_rec_loss_epoch / num_batches_train if num_batches_train > 0 else 0
    avg_ssl_loss = total_ssl_loss_epoch / num_batches_train if num_batches_train > 0 else 0

    model.eval()
    all_hits_eval = []
    all_mrrs_eval = []
    all_precisions_eval = []
    k_metric = 20

    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data[0] is None: continue
            pyg_batch, original_sequences_padded, attention_masks, targets, original_sequence_lens = batch_data
            if pyg_batch is not None and (pyg_batch.x is None or pyg_batch.x.numel() == 0) and original_sequences_padded.numel() == 0: continue

            if pyg_batch is not None: pyg_batch = pyg_batch.to(device)
            original_sequences_padded = original_sequences_padded.to(device)
            attention_masks = attention_masks.to(device)
            targets = targets.to(device).squeeze()
            original_sequence_lens = original_sequence_lens.to(device)


            scores = model(pyg_batch, original_sequences_padded, attention_masks, original_sequence_lens, is_train=False)
            if scores is None or scores.numel() == 0: continue

            _, top_k_indices_from_scores = scores.topk(k_metric, dim=1)
            top_k_item_ids = top_k_indices_from_scores + 1

            targets_cpu = targets.cpu().numpy()
            top_k_item_ids_cpu = top_k_item_ids.cpu().numpy()

            for i in range(targets_cpu.shape[0]):
                target_id = targets_cpu[i]
                predicted_ids = top_k_item_ids_cpu[i]
                if target_id == 0: continue

                is_hit = target_id in predicted_ids
                all_hits_eval.append(1 if is_hit else 0)
                if is_hit:
                    rank_list = np.where(predicted_ids == target_id)[0]
                    if len(rank_list) > 0: # اطمینان از اینکه آیتم پیدا شده
                        rank = rank_list[0] + 1
                        all_mrrs_eval.append(1.0 / rank)
                        all_precisions_eval.append(1.0 / k_metric)
                    else: # نباید اتفاق بیفتد اگر is_hit درست است
                        all_mrrs_eval.append(0.0)
                        all_precisions_eval.append(0.0)
                else:
                    all_mrrs_eval.append(0.0)
                    all_precisions_eval.append(0.0)

    recall_at_20_epoch = np.mean(all_hits_eval) * 100 if all_hits_eval else 0
    mrr_at_20_epoch = np.mean(all_mrrs_eval) * 100 if all_mrrs_eval else 0
    precision_at_20_epoch = np.mean(all_precisions_eval) * 100 if all_precisions_eval else 0

    print(f"Epoch {epoch+1}/{opt.epoch} | Avg Loss: {avg_total_loss:.4f} (Rec: {avg_rec_loss:.4f}, SSL: {avg_ssl_loss:.4f})")
    print(f"Eval: Recall@{k_metric}: {recall_at_20_epoch:.2f}% | MRR@{k_metric}: {mrr_at_20_epoch:.2f}% | Precision@{k_metric}: {precision_at_20_epoch:.2f}%")

    if recall_at_20_epoch > best_recall_at_20:
        best_recall_at_20 = recall_at_20_epoch
        save_path = os.path.join(checkpoint_dir, f"model_best_pyg_adv.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Best Model saved to {save_path} (Recall@{k_metric}: {best_recall_at_20:.2f}%)")

print("Training finished.")