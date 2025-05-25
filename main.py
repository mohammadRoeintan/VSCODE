import torch
from torch.optim import Adam
# از model، کلاس SessionGraph و تابع train_test (اگر هنوز استفاده می‌شود) را ایمپورت کنید
# اما با توجه به تغییرات، train_test ممکن است نیاز به بازنویسی داشته باشد یا منطقش به main منتقل شود.
# در اینجا، منطق حلقه آموزش و تست را در خود main نگه می‌داریم.
from model_pyg_corrected import SessionGraph, PositionalEncoding # فرض می‌کنیم مدل در این فایل است
from utils_pyg_corrected import generate_dataloader_pyg # فرض می‌کنیم utils در این فایل است
import numpy as np
import os
import pickle # برای بارگذاری داده‌های pickle شده

# --- پارامترها (می‌توانید از argparse مانند کد اصلی استفاده کنید) ---
# برای سادگی، مقادیر ثابت در نظر گرفته شده‌اند. شما باید argparse را بازگردانید.
class Opt: # یک کلاس ساده برای شبیه‌سازی opt
    dataset = 'sample_pyg' # نام دیتاست خود را بگذارید
    hiddenSize = 100
    batchSize = 128 # در generate_dataloader_pyg استفاده می‌شود
    epoch = 10
    lr = 0.001
    # پارامترهای GNN
    num_gnn_steps = 1 # تعداد لایه‌های GatedGraphConv
    # پارامترهای Transformer
    nhead_transformer = 2
    nlayers_transformer = 1
    dropout_transformer = 0.2
    # پارامتر SSL
    ssl_weight = 0.0 # فعلاً SSL را غیرفعال می‌کنیم تا روی بخش اصلی تمرکز کنیم
    # سایر پارامترها
    nonhybrid = False # یا True، بسته به منطق امتیازدهی شما
    n_node = 310 # مقدار پیش‌فرض برای sample, باید با داده شما تنظیم شود

opt = Opt()
# -------------------------------------------------------------

# --- بارگذاری داده‌ها (باید با فرمت pickle شما سازگار باشد) ---
# فرض می‌کنیم train.txt و test.txt شامل تاپل (list_of_sessions, list_of_targets) هستند
data_dir = "./datasets" # مسیر دیتاست خود را تنظیم کنید
try:
    with open(os.path.join(data_dir, opt.dataset, "train.txt"), 'rb') as f:
        train_sessions, train_targets = pickle.load(f)
    with open(os.path.join(data_dir, opt.dataset, "test.txt"), 'rb') as f:
        test_sessions, test_targets = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Data files not found in {os.path.join(data_dir, opt.dataset)}")
    print("Please create dummy train.txt and test.txt for 'sample_pyg' or provide correct dataset.")
    # ایجاد داده نمونه برای تست (اگر فایل‌ها موجود نیستند)
    print("Creating dummy data for 'sample_pyg'...")
    train_sessions = [[1,2,3], [2,3,4,2], [5,1], [6,7,8,9,6]]
    train_targets = [4,5,2,6]
    test_sessions = [[1,2], [3,4,5]]
    test_targets = [3,2]
    # به‌روزرسانی n_node بر اساس داده نمونه
    all_items = set(sum(train_sessions, []) + sum(test_sessions, []) + train_targets + test_targets)
    if all_items:
      opt.n_node = max(all_items) + 1
    else:
      opt.n_node = 10 # یک مقدار کوچک اگر داده خالی است
    print(f"Using dummy data with n_node = {opt.n_node}")

# -------------------------------------------------------------


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# اطمینان از اینکه n_node به درستی مقداردهی شده (باید بزرگتر از بیشترین ID آیتم باشد)
# این بخش باید با دقت بیشتری بر اساس داده‌های واقعی شما تنظیم شود.
# اگر از داده نمونه بالا استفاده می‌کنید، n_node باید حداقل max(all_items) + 1 باشد.
# opt.n_node = 310 # یا مقدار صحیح برای دیتاست شما

model = SessionGraph(
    n_node=opt.n_node,
    hidden_size=opt.hiddenSize,
    num_gnn_steps=opt.num_gnn_steps,
    ssl_weight=opt.ssl_weight,
    nhead_transformer=opt.nhead_transformer,
    nlayers_transformer=opt.nlayers_transformer,
    dropout_transformer=opt.dropout_transformer,
    nonhybrid=opt.nonhybrid
).to(device)

optimizer = Adam(model.parameters(), lr=opt.lr)

# ایجاد DataLoader با استفاده از تابع جدید
# (مطمئن شوید sessions و targets به درستی بارگذاری شده‌اند)
train_loader = generate_dataloader_pyg(train_sessions, train_targets, batch_size=opt.batchSize, shuffle=True, num_workers=0)
test_loader = generate_dataloader_pyg(test_sessions, test_targets, batch_size=opt.batchSize, shuffle=False, num_workers=0)


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

    for pyg_batch, original_sequences_padded, attention_masks, targets in train_loader:
        if pyg_batch.x is None or pyg_batch.x.numel() == 0 : # اگر بچ خالی از گراف معتبر باشد
            # print("Skipping empty batch in training.")
            continue

        pyg_batch = pyg_batch.to(device)
        original_sequences_padded = original_sequences_padded.to(device)
        attention_masks = attention_masks.to(device) # ماسک بولی
        targets = targets.to(device).squeeze() # (B)

        optimizer.zero_grad()
        # فراخوانی forward مدل با ورودی‌های جدید
        scores, ssl_loss = model(pyg_batch, original_sequences_padded, attention_masks, is_train=True)

        # محاسبه loss توصیه
        # scores: (B, n_node-1) , targets: (B) شامل IDهای اصلی از 1 تا n_node-1
        # تبدیل اهداف به اندیس 0پایه برای CrossEntropyLoss
        # اهداف باید در محدوده [0, num_classes-1] باشند. در اینجا num_classes = n_node-1 (چون پدینگ 0 را حذف می‌کنیم)
        # پس اهداف اصلی (1 تا N) باید به (0 تا N-1) نگاشت شوند.
        valid_targets_mask = (targets > 0) # فقط اهداف معتبر (نه پدینگ اگر در داده باشد)
        if valid_targets_mask.sum() == 0: # اگر هیچ هدف معتبری در بچ نیست
            # print("Skipping batch with no valid targets.")
            rec_loss_batch = torch.tensor(0.0, device=device)
        else:
            # اطمینان از اینکه ابعاد scores و targets[valid_targets_mask] برای loss سازگارند
            # scores باید برای تمام نمونه‌های بچ باشد، targets هم
            # CrossEntropyLoss انتظار دارد scores (B, C) و target (B) باشد
            # scores (B, n_node-1)
            # targets (B) شامل IDهای از 1 تا n_node
            # target_for_loss (B_valid) شامل IDهای از 0 تا n_node-2
            
            # فقط برای نمونه‌هایی که هدف معتبر دارند loss را محاسبه می‌کنیم
            # این یعنی scores هم باید برای همان نمونه‌ها فیلتر شود
            if scores[valid_targets_mask].shape[0] > 0 :
                 target_for_loss = (targets[valid_targets_mask] - 1).clamp(0, opt.n_node - 2) # -1 برای 0-پایه، -1 دیگر برای حذف پدینگ از n_node
                 rec_loss_batch = model.loss_function(scores[valid_targets_mask], target_for_loss)
            else: # اگر scores[valid_targets_mask] خالی شد (نباید اتفاق بیفتد اگر valid_targets_mask.sum() > 0)
                 rec_loss_batch = torch.tensor(0.0, device=device)


        # loss نهایی
        loss_batch = rec_loss_batch + ssl_loss # ssl_loss از قبل با وزن ضرب شده (اگر ssl_weight > 0)

        loss_batch.backward()
        optimizer.step()

        total_loss_epoch += loss_batch.item()
        total_rec_loss_epoch += rec_loss_batch.item()
        total_ssl_loss_epoch += ssl_loss.item() # ssl_loss از مدل با وزن می‌آید
        num_batches_train += 1

    avg_total_loss = total_loss_epoch / num_batches_train if num_batches_train > 0 else 0
    avg_rec_loss = total_rec_loss_epoch / num_batches_train if num_batches_train > 0 else 0
    avg_ssl_loss = total_ssl_loss_epoch / num_batches_train if num_batches_train > 0 else 0

    # ارزیابی مدل
    model.eval()
    all_hits_eval = []
    all_mrrs_eval = []
    all_precisions_eval = [] # برای Precision@k
    k_metric = 20

    with torch.no_grad():
        for pyg_batch, original_sequences_padded, attention_masks, targets in test_loader:
            if pyg_batch.x is None or pyg_batch.x.numel() == 0:
                # print("Skipping empty batch in testing.")
                continue

            pyg_batch = pyg_batch.to(device)
            original_sequences_padded = original_sequences_padded.to(device)
            attention_masks = attention_masks.to(device)
            targets = targets.to(device).squeeze() # (B)

            scores = model(pyg_batch, original_sequences_padded, attention_masks, is_train=False) # (B, n_node-1)

            # گرفتن k آیتم برتر
            # scores مربوط به آیتم‌های 1 تا n_node-1 است (اندیس 0 تا n_node-2)
            _, top_k_indices_from_scores = scores.topk(k_metric, dim=1) # (B, k) این اندیس‌ها 0-پایه هستند نسبت به scores

            # تبدیل اندیس‌های top_k به IDهای اصلی آیتم (1 تا n_node-1)
            top_k_item_ids = top_k_indices_from_scores + 1 # چون scores برای آیتم‌های 1-پایه است که 0-پایه شده‌اند

            targets_cpu = targets.cpu().numpy()
            top_k_item_ids_cpu = top_k_item_ids.cpu().numpy()

            for i in range(targets_cpu.shape[0]):
                target_id = targets_cpu[i]
                predicted_ids = top_k_item_ids_cpu[i]

                if target_id == 0: # نادیده گرفتن اهداف پدینگ (اگر وجود داشته باشند)
                    continue

                is_hit = target_id in predicted_ids
                all_hits_eval.append(1 if is_hit else 0)

                if is_hit:
                    rank_list = np.where(predicted_ids == target_id)[0]
                    rank = rank_list[0] + 1
                    all_mrrs_eval.append(1.0 / rank)
                    all_precisions_eval.append(1.0 / k_metric)
                else:
                    all_mrrs_eval.append(0.0)
                    all_precisions_eval.append(0.0)

    recall_at_20_epoch = np.mean(all_hits_eval) * 100 if all_hits_eval else 0
    mrr_at_20_epoch = np.mean(all_mrrs_eval) * 100 if all_mrrs_eval else 0
    precision_at_20_epoch = np.mean(all_precisions_eval) * 100 if all_precisions_eval else 0 # محاسبه صحیح Precision

    print(f"Epoch {epoch+1}/{opt.epoch} | Avg Loss: {avg_total_loss:.4f} (Rec: {avg_rec_loss:.4f}, SSL: {avg_ssl_loss:.4f})")
    print(f"Eval: Recall@{k_metric}: {recall_at_20_epoch:.2f}% | MRR@{k_metric}: {mrr_at_20_epoch:.2f}% | Precision@{k_metric}: {precision_at_20_epoch:.2f}%")

    if recall_at_20_epoch > best_recall_at_20:
        best_recall_at_20 = recall_at_20_epoch
        save_path = os.path.join(checkpoint_dir, f"model_best.pth") # ذخیره بهترین مدل
        torch.save(model.state_dict(), save_path)
        print(f"✅ Best Model saved to {save_path} (Recall@{k_metric}: {best_recall_at_20:.2f}%)")

    # ذخیره مدل هر اپوک (اختیاری)
    # save_path_epoch = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    # torch.save(model.state_dict(), save_path_epoch)
    # print(f"Model of epoch {epoch+1} saved to {save_path_epoch}")

print("Training finished.")