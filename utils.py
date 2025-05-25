import torch
import numpy as np
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch
import torch.nn.functional as F # برای F.pad

class PYGSessionDatasetFinal(torch.utils.data.Dataset):
    def __init__(self, sessions, targets):
        self.sessions = sessions
        self.targets = targets

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session_raw = self.sessions[idx]
        target = self.targets[idx]

        # IDهای اصلی آیتم‌ها در توالی (بدون فیلتر کردن 0 در اینجا، چون 0 padding_idx است)
        # اما برای گراف، فقط آیتم‌های > 0 را در نظر می‌گیریم
        original_sequence_ids = [int(i) for i in session_raw if isinstance(i, (int, float)) and not np.isnan(i)]
        if not original_sequence_ids: # اگر جلسه کاملا خالی یا نامعتبر بود
            original_sequence_ids = [0] # یک آیتم پدینگ برای جلوگیری از خطا

        # آیتم‌های معتبر برای ساخت گراف (بزرگتر از 0)
        graph_items = [item_id for item_id in original_sequence_ids if item_id > 0]

        if not graph_items:
            # اگر هیچ آیتم معتبری برای گراف نیست (مثلاً جلسه فقط [0,0,0] بوده)
            return PyGData(
                x_gnn_node_ids=torch.tensor([[0]], dtype=torch.long), # نود پدینگ برای GNN
                edge_index_gnn=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor([target], dtype=torch.long),
                original_sequence_ids=torch.tensor(original_sequence_ids, dtype=torch.long),
                # نگاشت از اندیس نود در گراف جلسه به اندیس آن در توالی اصلی
                # برای جلسات خالی، این پیچیده می‌شود. فعلاً خالی می‌گذاریم یا placeholder
                node_to_seq_idx_map=torch.empty((0), dtype=torch.long),
                is_valid_session=torch.tensor(False, dtype=torch.bool)
            )

        unique_graph_node_ids = sorted(list(set(graph_items)))
        node_id_to_graph_idx_map = {item_id: i for i, item_id in enumerate(unique_graph_node_ids)}

        x_gnn_node_ids = torch.tensor(unique_graph_node_ids, dtype=torch.long)

        edge_list_gnn = []
        for i in range(len(graph_items) - 1):
            src_original_id = graph_items[i]
            dst_original_id = graph_items[i+1]
            # IDها از قبل در unique_graph_node_ids هستند چون از graph_items آمده‌اند
            src_mapped_idx = node_id_to_graph_idx_map[src_original_id]
            dst_mapped_idx = node_id_to_graph_idx_map[dst_original_id]
            edge_list_gnn.append([src_mapped_idx, dst_mapped_idx])

        edge_index_gnn = torch.tensor(edge_list_gnn, dtype=torch.long).t().contiguous() if edge_list_gnn else torch.empty((2,0), dtype=torch.long)

        # ساخت نگاشت از اندیس نود در گراف (0 تا N_unique-1) به اندیس اولین وقوع آن در original_sequence_ids
        # این برای تابع map_gnn_to_sequence_for_transformer حیاتی است.
        # توجه: اگر یک آیتم چند بار در توالی تکرار شده، فقط اولین وقوع آن در اینجا برای ساخت گراف یکتا در نظر گرفته شده.
        # این ممکن است با منطق GNN کد اصلی شما متفاوت باشد اگر GNN اصلی شما تکرارها را متفاوت می‌دید.
        # GNN اصلی شما روی نودهای یکتا در بچ کار می‌کرد، اما گراف را از توالی می‌ساخت.
        # این بخش نیاز به دقت دارد.
        # فرض: original_sequence_ids IDهای اصلی را دارد. x_gnn_node_ids IDهای یکتای آن (بزرگتر از 0) است.
        # ما می‌خواهیم برای هر آیتم در original_sequence_ids، اگر در x_gnn_node_ids هست، اندیس محلی GNN آن را پیدا کنیم.
        # این کار در map_gnn_to_sequence_for_transformer انجام می‌شود.
        # node_to_seq_idx_map در اینجا شاید خیلی مفید نباشد، چون ما برعکس نیاز داریم:
        # از آیتم توالی به نمایش GNN آن.

        return PyGData(
            x_gnn_node_ids=x_gnn_node_ids,
            edge_index_gnn=edge_index_gnn,
            y=torch.tensor([target], dtype=torch.long),
            original_sequence_ids=torch.tensor(original_sequence_ids, dtype=torch.long),
            is_valid_session=torch.tensor(True, dtype=torch.bool)
            # node_to_seq_idx_map دیگر لازم نیست اگر نگاشت در مدل انجام شود
        )

def collate_pyg_final(data_list):
    # جدا کردن داده‌های معتبر برای بچ کردن با PyG
    valid_pyg_data_items = [d for d in data_list if d.is_valid_session.item()]
    
    if not valid_pyg_data_items:
        pyg_batch = None # یا یک بچ خالی ساختگی
    else:
        pyg_batch = PyGBatch.from_data_list(valid_pyg_data_items)

    # پد کردن original_sequence_ids برای تمام آیتم‌های data_list
    all_original_sequences = [d.original_sequence_ids for d in data_list]
    max_len_transformer = 0
    if all_original_sequences:
        non_empty_original_sequences = [s for s in all_original_sequences if s.numel() > 0]
        if non_empty_original_sequences:
            max_len_transformer = max(s.size(0) for s in non_empty_original_sequences)
    if max_len_transformer == 0: max_len_transformer = 1 # حداقل طول 1 برای پدینگ

    sequences_padded_for_transformer = []
    attention_masks_for_transformer = [] # True برای معتبر، False برای پدینگ

    for seq_ids in all_original_sequences:
        seq_actual_len = seq_ids.size(0)
        # اگر seq_ids به دلیل جلسه خالی، فقط [0] باشد، seq_actual_len=1 اما باید 0 در نظر گرفته شود
        if seq_actual_len == 1 and seq_ids[0].item() == 0 and not any(item_id > 0 for item_id in seq_ids): # اگر فقط پدینگ بود
            seq_actual_len = 0 # طول واقعی صفر است
            
        padding_needed = max_len_transformer - seq_actual_len
        current_padded_seq = F.pad(seq_ids, (0, padding_needed), value=0) # پدینگ با 0 (padding_idx)
        sequences_padded_for_transformer.append(current_padded_seq)

        current_mask = torch.cat([
            torch.ones(seq_actual_len, dtype=torch.bool),
            torch.zeros(padding_needed, dtype=torch.bool)
        ])
        attention_masks_for_transformer.append(current_mask)
    
    targets_all = torch.cat([d.y for d in data_list], dim=0)

    # original_sequence_lens برای محاسبه ht در compute_scores
    original_sequence_lens_all = torch.tensor([d.original_sequence_ids.size(0) if (d.original_sequence_ids.size(0) > 1 or (d.original_sequence_ids.size(0)==1 and d.original_sequence_ids[0].item()!=0)) else 0 for d in data_list], dtype=torch.long)


    return pyg_batch, \
           torch.stack(sequences_padded_for_transformer), \
           torch.stack(attention_masks_for_transformer), \
           targets_all, \
           original_sequence_lens_all


def generate_dataloader_pyg_final(sessions, targets, batch_size=100, shuffle=True, num_workers=0):
    dataset = PYGSessionDatasetFinal(sessions, targets)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pyg_final,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 and torch.cuda.is_available() else False
    )