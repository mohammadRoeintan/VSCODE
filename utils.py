import torch
import numpy as np
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch
import torch.nn.functional as F # برای F.pad

class PYGSessionDatasetAdvanced(torch.utils.data.Dataset):
    def __init__(self, sessions, targets):
        self.sessions = sessions
        self.targets = targets

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session_raw = self.sessions[idx]
        target = self.targets[idx]

        cleaned_items = [int(i) for i in session_raw if isinstance(i, (int, float)) and not np.isnan(i) and int(i) > 0]

        if not cleaned_items:
            # جلسه خالی یا فقط شامل آیتم‌های نامعتبر
            return PyGData(
                x=torch.tensor([[0]], dtype=torch.long), # نود پدینگ برای GNN
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor([target], dtype=torch.long),
                original_sequence_ids=torch.tensor([0], dtype=torch.long), # توالی اصلی پد شده برای Transformer
                num_valid_nodes_in_graph=torch.tensor(0, dtype=torch.long), # تعداد نودهای معتبر در گراف GNN
                original_sequence_len=torch.tensor(0, dtype=torch.long) # طول اصلی توالی قبل از پدینگ
            )

        unique_node_ids = sorted(list(set(cleaned_items)))
        node_id_map = {item_id: i for i, item_id in enumerate(unique_node_ids)}

        # x: IDهای اصلی آیتم‌های یکتا برای Embedding در GNN
        x_gnn = torch.tensor(unique_node_ids, dtype=torch.long) # .unsqueeze(1) اگر Embedding انتظار آن را دارد

        edge_list = []
        for i in range(len(cleaned_items) - 1):
            src_original_id = cleaned_items[i]
            dst_original_id = cleaned_items[i+1]
            if src_original_id in node_id_map and dst_original_id in node_id_map:
                src_mapped_idx = node_id_map[src_original_id]
                dst_mapped_idx = node_id_map[dst_original_id]
                edge_list.append([src_mapped_idx, dst_mapped_idx])

        if not edge_list:
            edge_index_gnn = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index_gnn = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # original_sequence_ids: IDهای اصلی آیتم‌ها در توالی، برای استفاده در Transformer
        original_sequence_ids = torch.tensor(cleaned_items, dtype=torch.long)

        return PyGData(
            x=x_gnn, # IDهای اصلی نودهای یکتا برای GNN
            edge_index=edge_index_gnn,
            y=torch.tensor([target], dtype=torch.long),
            original_sequence_ids=original_sequence_ids, # IDهای اصلی توالی برای Transformer
            num_valid_nodes_in_graph=torch.tensor(len(unique_node_ids), dtype=torch.long),
            original_sequence_len=torch.tensor(len(cleaned_items), dtype=torch.long)
        )

def collate_pyg_advanced(data_list):
    # جدا کردن گراف‌های معتبر (آنهایی که واقعاً جلسه معتبر دارند)
    valid_data_list = [d for d in data_list if d.num_valid_nodes_in_graph.item() > 0]
    # جدا کردن اهداف و طول توالی‌های اصلی برای تمام نمونه‌ها (حتی خالی‌ها)
    targets_all = torch.cat([d.y for d in data_list], dim=0)
    original_sequence_lens_all = torch.stack([d.original_sequence_len for d in data_list])

    if not valid_data_list: # اگر تمام جلسات در بچ خالی بودند
        # ایجاد یک بچ گراف خالی و توالی‌های پد شده خالی
        # ابعاد باید با تعداد کل آیتم‌ها در data_list (batch_size) همخوانی داشته باشد
        batch_size = len(data_list)
        # None برای pyg_batch می‌تواند نشانه‌ای برای مدل باشد که این بچ خالی است
        # یا یک pyg_batch کاملا خالی بسازید.
        # pyg_batch = PyGBatch.from_data_list([]) # ممکن است باعث خطا شود اگر برخی ویژگی‌ها نباشند
        # یک راه حل: یک گراف placeholder برای هر آیتم خالی ایجاد کنید (پیچیده)
        # ساده‌ترین راه: None برگردانیم و در حلقه آموزش مدیریت کنیم.
        # در اینجا، فرض می‌کنیم مدل می‌تواند None را مدیریت کند یا از این بچ صرف نظر می‌شود.
        return None, \
               torch.zeros((batch_size, 0), dtype=torch.long), \
               torch.zeros((batch_size, 0), dtype=torch.bool), \
               targets_all, \
               original_sequence_lens_all # برای مدیریت ht و a

    pyg_batch = PyGBatch.from_data_list(valid_data_list)

    # پد کردن original_sequence_ids برای تمام آیتم‌های data_list (حتی خالی‌ها)
    # برای آیتم‌های خالی، original_sequence_ids آنها [0] خواهد بود (از __getitem__)
    # یا اگر در __getitem__ برای خالی‌ها، original_sequence_ids=[] برگردانده شود:
    all_original_sequences = []
    max_len_transformer = 0
    for d in data_list:
        # اگر جلسه خالی بود، یک توالی با یک پدینگ [0] برمی‌گردانیم تا stack خطا ندهد
        # این 0 باید با padding_idx امبدینگ همخوانی داشته باشد
        if d.original_sequence_len.item() == 0:
            all_original_sequences.append(torch.tensor([0], dtype=torch.long))
        else:
            all_original_sequences.append(d.original_sequence_ids)
        if d.original_sequence_len.item() > max_len_transformer:
             max_len_transformer = d.original_sequence_len.item()
    
    # اگر همه توالی‌ها خالی بودند (بعد از پدینگ برای آیتم‌های خالی)، max_len_transformer می‌تواند 1 باشد
    if max_len_transformer == 0 and all_original_sequences : max_len_transformer = 1


    sequences_padded_for_transformer = []
    attention_masks_for_transformer = []

    for i, seq_ids in enumerate(all_original_sequences):
        seq_actual_len = original_sequence_lens_all[i].item() # طول واقعی قبل از پدینگ
        
        # پد کردن توالی با 0
        padding_needed = max_len_transformer - seq_actual_len
        if padding_needed < 0 : padding_needed = 0 # نباید اتفاق بیفتد اگر max_len درست باشد

        current_padded_seq = F.pad(seq_ids, (0, padding_needed), value=0)
        sequences_padded_for_transformer.append(current_padded_seq)

        # ایجاد ماسک (True برای آیتم‌های معتبر، False برای پدینگ)
        current_mask = torch.cat([
            torch.ones(seq_actual_len, dtype=torch.bool),
            torch.zeros(padding_needed, dtype=torch.bool)
        ])
        attention_masks_for_transformer.append(current_mask)

    if not sequences_padded_for_transformer: # اگر data_list خالی بود (نباید اتفاق بیفتد)
        return pyg_batch, \
               torch.empty((0,0), dtype=torch.long), \
               torch.empty((0,0), dtype=torch.bool), \
               targets_all, \
               original_sequence_lens_all

    return pyg_batch, \
           torch.stack(sequences_padded_for_transformer), \
           torch.stack(attention_masks_for_transformer), \
           targets_all, \
           original_sequence_lens_all


def generate_dataloader_pyg_advanced(sessions, targets, batch_size=100, shuffle=True, num_workers=0):
    dataset = PYGSessionDatasetAdvanced(sessions, targets)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pyg_advanced,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 and torch.cuda.is_available() else False
    )