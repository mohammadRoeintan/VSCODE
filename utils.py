import networkx as nx
import numpy as np
import torch
 

from scipy.sparse import lil_matrix, csr_matrix

def build_graph(train_data):
    # ابتدا بررسی می‌کنیم که آیا train_data یک تاپل است یا خیر
    if isinstance(train_data, tuple):
        # اگر تاپل بود، اولین عنصر آن را که لیست سشن‌هاست برمی‌داریم
        sessions = train_data[0]
    else:
        sessions = train_data
    
    max_node = max(max(seq) for seq in sessions) + 1
    adj_in = lil_matrix((max_node, max_node))
    adj_out = lil_matrix((max_node, max_node))
    
    for seq in sessions:
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            adj_out[u, v] += 1
            adj_in[v, u] += 1
    
    # نرمالایز کردن ماتریس‌ها
    adj_in = adj_in.tocsr()
    adj_out = adj_out.tocsr()
    for i in range(max_node):
        if adj_in[i].sum() > 0:
            adj_in[i] = adj_in[i] / adj_in[i].sum()
        if adj_out[i].sum() > 0:
            adj_out[i] = adj_out[i] / adj_out[i].sum()
    
    return adj_in, adj_out


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    
    # پد کردن با عدد 0 به جای لیست [0]
    us_pois = [upois + [item_tail] * (len_max - len(upois)) for upois in all_usr_pois]
    
    # حذف لیست‌های تودرتو (اگر وجود دارد)
    us_pois = [
        [item[0] if isinstance(item, list) else item 
        for item in session
    ] for session in us_pois]
    
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    
    # تبدیل به تانسور
    try:
        us_pois_tensor = torch.tensor(us_pois, dtype=torch.long)
        us_msks_tensor = torch.tensor(us_msks, dtype=torch.long)
    except Exception as e:
        print("Error in tensor conversion:")
        print("Sample us_pois:", us_pois[:2])
        print("Sample us_msks:", us_msks[:2])
        raise e
    
    return us_pois_tensor, us_msks_tensor, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
 
