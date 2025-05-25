import torch
import numpy as np
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch

class PYGSessionDataset(torch.utils.data.Dataset):
    def __init__(self, sessions, targets):
        self.sessions = sessions
        self.targets = targets
        # برای جلوگیری از پردازش مکرر، می‌توان جلسات را یکبار اینجا به فرمت PyGData تبدیل کرد
        # اما اگر حافظه زیادی مصرف می‌کند، همان پردازش در __getitem__ بهتر است.
        # در اینجا فرض می‌کنیم پردازش در __getitem__ انجام می‌شود.

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session_raw = self.sessions[idx]
        target = self.targets[idx]

        # پاکسازی و استخراج آیتم‌های معتبر (بزرگتر از 0)
        cleaned_items = [int(i) for i in session_raw if isinstance(i, (int, float)) and not np.isnan(i) and int(i) > 0]

        if not cleaned_items:
            # اگر جلسه خالی یا فقط شامل آیتم‌های نامعتبر است
            # یک گراف placeholder با یک نود پدینگ (ID=0) برمی‌گردانیم
            # x: ویژگی نودها (ID آیتم‌ها)
            # edge_index: لیست یال‌ها
            # y: هدف
            # original_sequence: توالی اصلی (خالی)
            # num_valid_nodes: تعداد نودهای معتبر (0)
            return PyGData(
                x=torch.tensor([[0]], dtype=torch.long), # نود پدینگ
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor([target], dtype=torch.long), # هدف همچنان وجود دارد
                original_sequence=torch.tensor([], dtype=torch.long),
                num_valid_nodes=torch.tensor(0, dtype=torch.long) # برای تشخیص جلسات خالی
            )

        # نودهای یکتا و نگاشت آنها به اندیس‌های 0 تا N-1
        # node_ids شامل IDهای اصلی آیتم‌های یکتا در جلسه است
        unique_node_ids = sorted(list(set(cleaned_items))) # مرتب‌سازی برای ترتیب یکسان
        node_id_map = {item_id: i for i, item_id in enumerate(unique_node_ids)}

        # x: ویژگی نودها (IDهای اصلی آیتم‌های یکتا) برای Embedding
        x = torch.tensor(unique_node_ids, dtype=torch.long).unsqueeze(1) # شکل (num_unique_nodes, 1)

        # edge_index: یال‌ها بر اساس اندیس‌های محلی در unique_node_ids
        edge_list = []
        for i in range(len(cleaned_items) - 1):
            # cleaned_items[i] و cleaned_items[i+1] IDهای اصلی هستند
            # آنها را به اندیس‌های محلی نگاشت می‌کنیم
            src_original_id = cleaned_items[i]
            dst_original_id = cleaned_items[i+1]
            # فقط اگر هر دو نود در unique_node_ids باشند (باید باشند چون از cleaned_items آمده‌اند)
            if src_original_id in node_id_map and dst_original_id in node_id_map:
                src_mapped_idx = node_id_map[src_original_id]
                dst_mapped_idx = node_id_map[dst_original_id]
                edge_list.append([src_mapped_idx, dst_mapped_idx])

        if not edge_list: # اگر جلسه تک آیتمی باشد و یالی وجود نداشته باشد
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # original_sequence: توالی اصلی آیتم‌ها (IDهای اصلی) برای بخش Transformer
        # این باید به یک طول ثابت پد شود یا در collate_fn مدیریت شود.
        # در اینجا فقط خود توالی را برمی‌گردانیم و پدینگ را به collate_fn یا مدل می‌سپاریم.
        # یا می‌توانیم از قبل پد کنیم.
        # برای سادگی فعلی، خود توالی را برمی‌گردانیم.
        original_sequence_tensor = torch.tensor(cleaned_items, dtype=torch.long)

        return PyGData(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([target], dtype=torch.long),
            original_sequence=original_sequence_tensor, # IDهای اصلی آیتم‌های توالی
            num_valid_nodes=torch.tensor(len(unique_node_ids), dtype=torch.long)
        )

def collate_pyg(data_list):
    """
    تابع Collate سفارشی برای پد کردن original_sequence و بچ کردن گراف‌ها.
    """
    # جدا کردن گراف‌ها از بقیه داده‌ها
    batch_pyg = PyGBatch.from_data_list([d for d in data_list if d.num_valid_nodes.item() > 0])
    
    # اگر تمام گراف‌ها خالی بودند، یک بچ خالی برگردان
    if not batch_pyg.x.numel(): # بررسی اینکه آیا batch_pyg.x خالی است یا نه
        # ایجاد یک بچ ساختگی خالی با ساختار مورد انتظار
        # این بخش نیاز به تعریف دقیق‌تری از ساختار بچ خالی دارد
        # مثلاً، yها همچنان می‌توانند وجود داشته باشند.
        ys = torch.cat([d.y for d in data_list], dim=0)
        # برای original_sequences_padded و masks هم باید مقادیر خالی مناسب برگردانده شود
        # یا اینکه از این بچ‌ها در حلقه آموزش صرف نظر شود.
        # فعلاً یک بچ گراف خالی و yها را برمی‌گردانیم.
        # بقیه فیلدها را باید در مدل مدیریت کرد.
        return PyGBatch.from_data_list([]), \
               torch.zeros((len(data_list), 0), dtype=torch.long), \
               torch.zeros((len(data_list), 0), dtype=torch.bool), \
               ys

    # پد کردن original_sequenceها
    original_sequences = [d.original_sequence for d in data_list]
    max_len = 0
    if original_sequences: # فقط اگر لیستی از توالی‌ها وجود دارد
        # فیلتر کردن توالی‌های خالی که ممکن است از جلسات خالی آمده باشند
        non_empty_sequences = [s for s in original_sequences if s.numel() > 0]
        if non_empty_sequences:
             max_len = max(s.size(0) for s in non_empty_sequences)
        # else: max_len می‌ماند 0

    sequences_padded = []
    masks = []
    for seq in original_sequences:
        seq_len = seq.size(0)
        padding_len = max_len - seq_len
        if padding_len >= 0 : # اگر max_len بزرگتر یا مساوی طول توالی باشد
            sequences_padded.append(F.pad(seq, (0, padding_len), value=0)) # پدینگ با 0
            masks.append(torch.cat([torch.ones(seq_len, dtype=torch.bool), torch.zeros(padding_len, dtype=torch.bool)]))
        else: # این حالت نباید رخ دهد اگر max_len درست محاسبه شده باشد
            sequences_padded.append(seq) # یا خطا
            masks.append(torch.ones(seq_len, dtype=torch.bool))


    if not sequences_padded: # اگر همه توالی‌ها خالی بودند
        original_sequences_padded = torch.empty((len(data_list), 0), dtype=torch.long)
        attention_masks = torch.empty((len(data_list), 0), dtype=torch.bool)
    else:
        original_sequences_padded = torch.stack(sequences_padded)
        attention_masks = torch.stack(masks)

    # استخراج اهداف (y)
    ys = torch.cat([d.y for d in data_list], dim=0)

    return batch_pyg, original_sequences_padded, attention_masks, ys


def generate_dataloader_pyg(sessions, targets, batch_size=100, shuffle=True, num_workers=0):
    dataset = PYGSessionDataset(sessions, targets)
    # استفاده از collate_fn سفارشی
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pyg, # استفاده از تابع collate جدید
        num_workers=num_workers, # برای بارگذاری موازی داده‌ها (اگر CPU bottleneck است)
        pin_memory=True if num_workers > 0 else False # برای انتقال سریعتر به GPU
    )