import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import * # model وارد شده است
import torch # torch اضافه شد برای بررسی availability CUDA

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

# -------------- پارامترهای جدید برای SSL --------------
parser.add_argument('--ssl_weight', type=float, default=0.1, help='Weight for Self-Supervised Learning Loss')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='Temperature parameter for InfoNCE Loss')
parser.add_argument('--ssl_dropout_rate', type=float, default=0.2, help='Dropout rate for SSL augmentation')
# ----------------------------------------------------

opt = parser.parse_args()
print(opt)

# اضافه کردن بررسی CUDA
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


def main():
    # مسیر دیتاست‌ها را مطابق با ساختار خودتان تنظیم کنید
    dataset_path = '/kaggle/working/VSCODE/datasets/' # مسیر پایه دیتاست

    train_data_file = dataset_path + opt.dataset + '/train.txt'
    test_data_file = dataset_path + opt.dataset + '/test.txt'

    print(f"Loading training data from: {train_data_file}")
    try:
        train_data = pickle.load(open(train_data_file, 'rb'))
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_file}")
        return
    except Exception as e:
        print(f"Error loading training data: {e}")
        return


    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        print(f"Loading testing data from: {test_data_file}")
        try:
            test_data = pickle.load(open(test_data_file, 'rb'))
        except FileNotFoundError:
            print(f"Error: Testing data file not found at {test_data_file}")
            # اگر فایل تست الزامی نیست، می‌توانید ادامه دهید یا خطا دهید
            # return
            # فرض می‌کنیم اگر فایل تست نباشد و validation هم نباشد، خطا می‌دهیم
            if not opt.validation:
                 print("Error: Test data not found and validation is not enabled.")
                 return
        except Exception as e:
             print(f"Error loading testing data: {e}")
             return


    # --- پردازش train_data برای ساخت گراف ---
    if isinstance(train_data, tuple):
        sessions_for_graph = train_data[0]
    else:
        # اگر train_data فقط لیست session ها باشد (مثلا در حالت بدون validation)
        # نیاز به بررسی فرمت دقیق train_data و test_data داریم
        # فرض می‌کنیم train_data همیشه فرمت (sessions, targets) دارد
        # اگر فرمت متفاوت است، این بخش نیاز به اصلاح دارد
        print("Warning: Unexpected format for train_data when not using validation. Assuming it contains only sessions.")
        sessions_for_graph = train_data # این فرض ممکن است نیاز به بازبینی داشته باشد

    # تبدیل آیتم‌های لیست به اعداد صحیح در sessions_for_graph
    if sessions_for_graph: # اطمینان از خالی نبودن
        sessions_for_graph = [
            [
                int(item[0]) if isinstance(item, list) and item else (int(item) if not isinstance(item, list) else 0) # اضافه کردن بررسی خالی بودن item
                for item in session if item is not None # اضافه کردن بررسی None
            ]
            for session in sessions_for_graph if session # اطمینان از خالی نبودن session
        ]
        # حذف session های خالی احتمالی بعد از پردازش
        sessions_for_graph = [session for session in sessions_for_graph if session]
        # ساخت گراف فقط در صورتی که session معتبر وجود داشته باشد
        if sessions_for_graph:
            adj_in, adj_out = build_graph(sessions_for_graph)
            print("Graph built successfully.")
        else:
            print("Error: No valid sessions found to build the graph.")
            return
    else:
        print("Error: sessions_for_graph is empty.")
        return
    # -----------------------------------------

    # ایجاد شیء Data
    try:
        # train_data باید فرمت (sessions, targets) داشته باشد
        if not isinstance(train_data, tuple) or len(train_data) != 2:
             raise ValueError("train_data must be a tuple of (sessions, targets)")
        # اطمینان از اینکه targets وجود دارد
        if not train_data[1]:
             raise ValueError("targets in train_data are missing or empty")

        train_data_obj = Data(train_data, shuffle=True)
        print("Training Data object created successfully.")
    except Exception as e:
        print(f"Error creating Training Data object: {e}")
        # نمایش نمونه داده‌ها برای دیباگ
        if isinstance(train_data, tuple) and len(train_data) == 2:
            print("Sample session:", train_data[0][0] if train_data[0] else "None")
            print("Sample target:", train_data[1][0] if train_data[1] else "None")
        raise e

    try:
        if not isinstance(test_data, tuple) or len(test_data) != 2:
             raise ValueError("test_data must be a tuple of (sessions, targets)")
        if not test_data[1]:
             raise ValueError("targets in test_data are missing or empty")

        test_data_obj = Data(test_data, shuffle=False)
        print("Testing Data object created successfully.")
    except Exception as e:
        print(f"Error creating Testing Data object: {e}")
        if isinstance(test_data, tuple) and len(test_data) == 2:
             print("Sample session:", test_data[0][0] if test_data[0] else "None")
             print("Sample target:", test_data[1][0] if test_data[1] else "None")
        raise e


    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        # برای دیتاست sample یا هر دیتاست دیگری که ممکن است اضافه شود
        # محاسبه n_node بر اساس داده‌های واقعی بهتر است
        all_nodes = set()
        if isinstance(train_data, tuple):
            for session in train_data[0]:
                all_nodes.update(item for item in session if item is not None and item != 0) # اضافه کردن بررسی None و 0
        if isinstance(test_data, tuple):
             for session in test_data[0]:
                all_nodes.update(item for item in session if item is not None and item != 0)

        if not all_nodes:
             print("Warning: No nodes found in the dataset. Setting n_node to default 310.")
             n_node = 310 # مقدار پیش‌فرض اگر هیچ نودی یافت نشد
        else:
             n_node = max(all_nodes) + 1 # +1 چون اندیس‌ها از 0 شروع می‌شوند (یا 1 اگر 0 پدینگ باشد)
        print(f"Calculated n_node based on data: {n_node}")


    # اطمینان از اینکه n_node به درستی محاسبه شده
    if n_node <= 1:
        print(f"Error: Invalid n_node calculated: {n_node}. Check dataset processing.")
        return

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        # train_test حالا باید train_data_obj و test_data_obj را بگیرد
        hit, mrr = train_test(model, train_data_obj, test_data_obj, opt) # ارسال opt به train_test
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