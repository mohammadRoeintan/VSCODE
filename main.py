import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')  # کم شد برای سریع‌تر شدن
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='steps after which lr decays')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propagation steps')
parser.add_argument('--patience', type=int, default=5, help='epochs to wait before early stop')  # کم شد برای سریع‌تر شدن
parser.add_argument('--nonhybrid', action='store_true', help='only use global preference to predict')
parser.add_argument('--validation', action='store_true', help='use validation set')
parser.add_argument('--valid_portion', type=float, default=0.1, help='portion of training as validation')

# اضافه کردن پارامتر drop_rate برای تیونینگ
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate for regularization')

opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open('/kaggle/working/VSCODE/datasets/' + opt.dataset + '/train.txt', 'rb'))

    # محدود کردن به ۵٪ داده
    if isinstance(train_data, tuple):
        sessions, targets = train_data
        num_samples = int(len(sessions) * 0.05)
        sessions = sessions[:num_samples]
        targets = targets[:num_samples]
        train_data = (sessions, targets)
    else:
        num_samples = int(len(train_data) * 0.05)
        train_data = train_data[:num_samples]
    print(f"Train data reduced to {num_samples} samples.")

    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('/kaggle/working/VSCODE/datasets/' + opt.dataset + '/test.txt', 'rb'))

        # محدود کردن تست هم به ۵٪
        if isinstance(test_data, tuple):
            sessions, targets = test_data
            num_samples = int(len(sessions) * 0.05)
            sessions = sessions[:num_samples]
            targets = targets[:num_samples]
            test_data = (sessions, targets)
        else:
            num_samples = int(len(test_data) * 0.05)
            test_data = test_data[:num_samples]
        print(f"Test data reduced to {num_samples} samples.")

    # استخراج sessionها و targetها
    if isinstance(train_data, tuple):
        sessions, targets = train_data
        sessions = [
            [int(item[0]) if isinstance(item, list) else int(item) for item in session]
            for session in sessions
        ]
    else:
        sessions = train_data

    adj_in, adj_out = build_graph(sessions)

    try:
        train_data = Data((sessions, targets), shuffle=True)
        print("Data object created successfully")
    except Exception as e:
        print("Error creating Data object:")
        print("Sample session:", sessions[0])
        print("Sample target:", targets[0] if len(targets) > 0 else "None")
        raise e

    test_data = Data(test_data, shuffle=False)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    # پاس دادن opt با پارامترهای کامل (مثلاً dropout) به مدل
    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()
