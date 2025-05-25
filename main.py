import torch
from torch.optim import Adam
from model_pyg_final_attempt import SessionGraph # نام فایل مدل جدید
from utils_pyg_final_attempt import generate_dataloader_pyg_final # نام فایل utils جدید
import numpy as np
import os
import pickle
import argparse # بازگرداندن argparse

# --- آرگومان‌ها (از کد اصلی شما) ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample_final', help='dataset name')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size') # کوچکتر برای تست
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=5, help='the number of epochs to train for') # کمتر برای تست
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps (num_gnn_steps)')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
# --data_subset_ratio و validation از کد اصلی شما می‌توانند اضافه شوند
parser.add_argument('--ssl_weight', type=float, default=0.0, help='Weight for Self-Supervised Learning Loss (فعلا غیرفعال)')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='Temperature parameter for InfoNCE Loss')
parser.add_argument('--ssl_dropout_rate', type=float, default=0.2, help='Dropout rate for SSL augmentation')
parser.add_argument('--nhead', type=int, default=2, help='number of heads in transformer encoder')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers in transformer encoder')
parser.add_argument('--ff_hidden', type=int, default=256, help='dimension of feedforward network model in transformer')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in transformer/positional encoding')
opt = parser.parse_args()
# ------------------------------------------

# --- بارگذاری داده‌ها ---
data_dir = "./datasets"
train_sessions, train_targets = [], []
test_sessions, test_targets = [], []
n_node_auto = 20 # مقدار پیش‌فرض اگر دیتاست یافت نشد
try:
    with open(os.path.join(data_dir, opt.dataset, "train.txt"), 'rb') as f:
        train_sessions, train_targets = pickle.load(f)
    with open(os.path.join(data_dir, opt.dataset, "test.txt"), 'rb') as f:
        test_sessions, test_targets = pickle.load(f)
    
    all_items_for_n_node = set()
    for sess_list in [train_sessions, test_sessions]:
        for sess in sess_list:
            for item in sess:
                if isinstance(item, (int, float)) and not np.isnan(item) and int(item) > 0:
                    all_items_for_n_node.add(int(item))
    for target_list in [train_targets, test_targets]:
        for target_item in target_list:
            if isinstance(target_item, (int, float)) and not np.isnan(target_item) and int(target_item) > 0:
                all_items_for_n_node.add(int(target_item))
    if all_items_for_n_node:
        n_node_auto = max(all_items_for_n_node) + 1

except FileNotFoundError:
    print(f"Warning: Data files not found for dataset '{opt.dataset}'. Creating dummy data...")
    train_sessions = [[1,2,3,4], [2,3,4,2,5], [5,1,6], [6,7,8,9,6,10], [11,12],[1],[1,2], [0,0]]
    train_targets = [5,1,7,11,1,2,3,1] # هدف برای جلسه پدینگ
    test_sessions = [[1,2,3], [3,4,5,6],[7], [0]]
    test_targets = [4,2,8,1]
    all_items = set(sum(train_sessions, []) + sum(test_sessions, []) + train_targets + test_targets)
    if all_items: n_node_auto = max(item for item in all_items if item > 0) + 1 if any(item > 0 for item in all_items) else 20
    else: n_node_auto = 20
    print(f"Using dummy data with n_node = {n_node_auto}")

opt.n_node = n_node_auto # استفاده از n_node محاسبه شده یا پیش‌فرض
# -------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if opt.n_node <=1 :
    print("Error: n_node must be > 1.")
    exit()

model = SessionGraph(opt, n_node=opt.n_node).to(device)
optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2 if hasattr(opt, 'l2') else 1e-5)
# Scheduler را هم می‌توانید از opt بخوانید
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step if hasattr(opt, 'lr_dc_step') else 3, gamma=opt.lr_dc if hasattr(opt, 'lr_dc') else 0.1)


train_loader = generate_dataloader_pyg_final(train_sessions, train_targets, batch_size=opt.batchSize, shuffle=True)
test_loader = generate_dataloader_pyg_final(test_sessions, test_targets, batch_size=opt.batchSize, shuffle=False)

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
        pyg_batch, original_sequences_padded, attention_masks, targets, original_sequence_lens = batch_data
        
        if original_sequences_padded.numel() == 0 : # اگر بچ کاملا خالی باشد
             # print("Skipping completely empty batch in training (no sequences).")
             continue
        if pyg_batch is None and torch.all(original_sequence_lens == 0): # اگر هیچ گراف معتبری نیست و همه توالی‌ها خالی‌اند
             # print("Skipping batch with no valid graphs and all empty sequences.")
             continue


        if pyg_batch is not None: pyg_batch = pyg_batch.to(device) # ممکن است None باشد
        original_sequences_padded = original_sequences_padded.to(device)
        attention_masks = attention_masks.to(device)
        targets = targets.to(device).squeeze()
        original_sequence_lens = original_sequence_lens.to(device)

        optimizer.zero_grad(set_to_none=True)
        scores, ssl_loss = model(pyg_batch, original_sequences_padded, attention_masks, original_sequence_lens, is_train=True)

        if scores is None or scores.numel() == 0:
            # print("Skipping batch due to empty scores from model.")
            continue

        valid_targets_mask = (targets > 0) # اهداف معتبر (ID > 0)
        num_valid_targets = valid_targets_mask.sum().item()

        rec_loss_batch = torch.tensor(0.0, device=device)
        if num_valid_targets > 0:
            current_scores = scores[valid_targets_mask]
            current_targets_for_loss = (targets[valid_targets_mask] - 1).clamp(0, opt.n_node - 2) # -1 برای 0پایه، -1 چون scores برای n_node-1 کلاس است
            
            if current_scores.shape[0] > 0 and current_scores.shape[0] == current_targets_for_loss.shape[0]:
                 if current_scores.shape[1] == opt.n_node -1 : # بررسی تعداد کلاس‌ها
                    rec_loss_batch = model.loss_function(current_scores, current_targets_for_loss)
                 # else:
                     # print(f"Score class dimension mismatch: {current_scores.shape[1]} vs {opt.n_node-1}")
            # else:
                # print(f"Shape mismatch for loss: scores_f {current_scores.shape}, targets_f {current_targets_for_loss.shape}")


        loss_batch = rec_loss_batch + ssl_loss
        if torch.isnan(loss_batch) or torch.isinf(loss_batch) or loss_batch.item() == 0 and rec_loss_batch.item() == 0 and num_valid_targets > 0 : # بررسی loss صفر مشکوک
            # print(f"Warning: Potentially problematic loss: {loss_batch.item()}. Rec: {rec_loss_batch.item()}, SSL: {ssl_loss.item()}")
            if torch.isnan(loss_batch) or torch.isinf(loss_batch): continue


        loss_batch.backward()
        optimizer.step()

        total_loss_epoch += loss_batch.item()
        if rec_loss_batch is not None: total_rec_loss_epoch += rec_loss_batch.item()
        if ssl_loss is not None: total_ssl_loss_epoch += ssl_loss.item()
        num_batches_train += 1
    
    scheduler.step() # آپدیت نرخ یادگیری

    avg_total_loss = total_loss_epoch / num_batches_train if num_batches_train > 0 else 0
    avg_rec_loss = total_rec_loss_epoch / num_batches_train if num_batches_train > 0 else 0
    avg_ssl_loss = total_ssl_loss_epoch / num_batches_train if num_batches_train > 0 else 0

    model.eval()
    all_hits_eval, all_mrrs_eval, all_precisions_eval = [], [], []
    k_metric = 20

    with torch.no_grad():
        for batch_data in test_loader:
            pyg_batch, original_sequences_padded, attention_masks, targets, original_sequence_lens = batch_data
            if original_sequences_padded.numel() == 0 : continue
            if pyg_batch is None and torch.all(original_sequence_lens == 0): continue

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
                    if len(rank_list) > 0:
                        rank = rank_list[0] + 1
                        all_mrrs_eval.append(1.0 / rank)
                        all_precisions_eval.append(1.0 / k_metric)
                    else:
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
        save_path = os.path.join(checkpoint_dir, f"model_best_pyg_final.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Best Model saved to {save_path} (Recall@{k_metric}: {best_recall_at_20:.2f}%)")

print("Training finished.")

if __name__ == '__main__':
    main()