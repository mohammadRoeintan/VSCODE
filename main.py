import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('/kaggle/working/TAGNN/datasets/' + opt.dataset + '/train.txt', 'rb'))
    print("Structure of loaded data:", type(train_data))  # برای دیباگ
    
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('/kaggle/working/TAGNN/datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    
    # اگر داده به صورت تاپل بارگذاری شده باشد
    # استخراج سشن‌ها و تارگت‌ها
    if isinstance(train_data, tuple):
        print("Data is a tuple, extracting sessions and targets...")
        sessions, targets = train_data
        print(f"Number of sessions: {len(sessions)}")
        print(f"Number of targets: {len(targets)}")
        
        # بررسی و اصلاح ساختار سشن‌ها
        cleaned_sessions = []
        for session in sessions:
            cleaned_session = []
            for item in session:
                if isinstance(item, list):
                    cleaned_session.append(item[0])
                else:
                    cleaned_session.append(item)
            cleaned_sessions.append(cleaned_session)
        sessions = cleaned_sessions
    else:
        sessions = train_data
        targets = [...]  # اگر تارگت‌ها جداگانه هستند
    
    # ساخت گراف
    adj_in, adj_out = build_graph(sessions)
    
    # ایجاد شیء Data
    try:
        train_data = Data((sessions, targets), shuffle=True)
        print("Data object created successfully")
    except Exception as e:
        print("Error creating Data object:")
        print("Sample session:", sessions[0])
        print("Sample target:", targets[0] if len(targets) > 0 else "None")
        raise e
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

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
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
