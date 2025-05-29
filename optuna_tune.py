import optuna
import subprocess
import sys
import os
import re # برای استخراج متریک با عبارت منظم

# --- تنظیمات اولیه ---
PYTHON_EXECUTABLE = sys.executable # مسیر پایتون فعلی
MAIN_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "main.py") # مسیر main.py
DATASET_NAME = "diginetica"  # نام دیتاست خود را اینجا وارد کنید
K_METRIC_VALUE = 20      # مقدار K برای متریک ها (مطابق با main.py)
N_TRIALS_OPTUNA = 30     # تعداد کل آزمایش های Optuna
USE_VALIDATION = True    # آیا از مجموعه اعتبارسنجی استفاده شود (برای انتخاب بهترین مدل)

# پارامترهایی که در هر آزمایش Optuna ثابت هستند اما به main.py پاس داده می شوند
# می توانید این لیست را بر اساس نیاز خود تغییر دهید
FIXED_ARGS_FOR_MAIN_PY = [
    "--dataset", DATASET_NAME,
    "--k_metric", str(K_METRIC_VALUE),
    "--patience", "5", # صبوری کمتر برای آزمایش های هایپرپارامتر
]
if USE_VALIDATION:
    FIXED_ARGS_FOR_MAIN_PY.append("--validation")


def parse_metric_from_output(output_stdout, metric_name="MRR"):
    """
    این تابع خروجی استاندارد main.py را برای استخراج متریک مورد نظر جستجو می کند.
    مثال خط مورد انتظار:
    Epoch End Evaluation @20 on Validation: Recall: X.XXXX%, MRR: Y.YYYY%
    یا اگر از --validation استفاده نشود:
    Epoch End Evaluation @20 on Test (during training): Recall: X.XXXX%, MRR: Y.YYYY%
    """
    # استفاده از عبارت منظم برای یافتن خط شامل متریک و استخراج مقدار آن
    # این الگو به دنبال خطی است که با "Epoch End Evaluation" شروع شده
    # و شامل "Validation" یا "Test (during training)" و سپس نام متریک (MRR یا Recall) باشد
    pattern_str = rf"Epoch End Evaluation @\d+ on (?:Validation|Test \(during training\)):.*{metric_name}:\s*([\d.]+)"
    match = re.search(pattern_str, output_stdout)

    if match:
        try:
            metric_value = float(match.group(1))
            print(f"Successfully parsed {metric_name}: {metric_value}")
            return metric_value
        except ValueError:
            print(f"Error: Could not convert parsed {metric_name} value to float: {match.group(1)}")
            return None
    else:
        # اگر الگوی اول پیدا نشد، سعی کنید فرمت دیگری را جستجو کنید که ممکن است در خروجی باشد
        # مانند گزارش های بهترین مدل در انتهای لاگ های main.py
        # Final Best Overall Result on Validation Set (k=20):
        # 	Recall@20: X.XXXX (Achieved at Epoch Y)
        # 	MRR@20: Z.ZZZZ (Achieved at Epoch W)
        if "Final Best Overall Result" in output_stdout:
             final_pattern_str = rf"\t{metric_name}@{K_METRIC_VALUE}:\s*([\d.]+)"
             final_match = re.search(final_pattern_str, output_stdout)
             if final_match:
                 try:
                     metric_value = float(final_match.group(1))
                     print(f"Successfully parsed final {metric_name}: {metric_value}")
                     return metric_value
                 except ValueError:
                    print(f"Error: Could not convert parsed final {metric_name} value to float: {final_match.group(1)}")
                    return None

        print(f"Warning: {metric_name} not found in the script output. Output was:\n{output_stdout[-1000:]}") # چاپ ۱۰۰۰ کاراکتر آخر خروجی برای دیباگ
        return None

def objective(trial: optuna.Trial):
    """
    تابع هدف برای Optuna.
    این تابع یک آزمایش (trial) را با هایپرپارامترهای پیشنهادی اجرا می کند.
    """
    # 1. پیشنهاد هایپرپارامترها توسط Optuna
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hiddenSize", [64, 100, 128, 256]) # مطابق با نیاز تغییر دهید
    batch_size = trial.suggest_categorical("batchSize", [ 128, 150 , 256]) # اگر حافظه GPU اجازه می دهد
    
    # پارامترهای SSL
    ssl_weight = trial.suggest_float("ssl_weight", 0.05, 0.3, log=True) # اگر از SSL استفاده می کنید
    # ssl_temp = trial.suggest_float("ssl_temp", 0.1, 1.0)
    # ssl_dropout_rate = trial.suggest_float("ssl_dropout_rate", 0.1, 0.5)

    # پارامترهای Transformer
    nhead = trial.suggest_categorical("nhead", [2, 4])
    nlayers = trial.suggest_categorical("nlayers", [1, 2])
    ff_hidden = trial.suggest_categorical("ff_hidden", [128, 256, 512])
    # dropout_transformer = trial.suggest_float("dropout", 0.1, 0.3) # توجه: main.py یک آرگومان dropout کلی دارد

    # پارامترهای GNN
    step_gnn = trial.suggest_int("step", 1, 2) # قدم های انتشار GNN محلی
    l2_penalty = trial.suggest_float("l2", 1e-6, 1e-4, log=True)
    global_gcn_layers = trial.suggest_categorical("global_gcn_layers", [0, 1, 2]) # 0 برای غیرفعال کردن

    # بودجه برای این آزمایش (برای هرس سریعتر)
    # Pruner از این مقادیر برای تصمیم گیری استفاده می کند
    # شما می توانید تعداد epoch ها را به عنوان منبع (resource) در نظر بگیرید
    epochs_for_this_trial = trial.suggest_int("epochs_for_trial", 5, 10) # تعداد کمتر برای آزمایش های اولیه سریع
    data_subset_for_this_trial = 0.05 # یا 0.1 - استفاده از ۵٪ یا ۱۰٪ داده ها

    # 2. ساخت دستور برای اجرای main.py
    command = [
        PYTHON_EXECUTABLE, MAIN_SCRIPT_PATH,
        "--lr", str(lr),
        "--hiddenSize", str(hidden_size),
        "--batchSize", str(batch_size),
        "--ssl_weight", str(ssl_weight),
        # "--ssl_temp", str(ssl_temp),
        # "--ssl_dropout_rate", str(ssl_dropout_rate),
        "--nhead", str(nhead),
        "--nlayers", str(nlayers),
        # "--ff_hidden", str(ff_hidden),
        # "--dropout", str(dropout_transformer), # اگر می خواهید dropout ترنسفورمر را جداگانه تنظیم کنید
        "--step", str(step_gnn),
        "--l2", str(l2_penalty),
        "--global_gcn_layers", str(global_gcn_layers),
        "--epoch", str(epochs_for_this_trial),
        "--data_subset_ratio", str(data_subset_for_this_trial),
    ]
    command.extend(FIXED_ARGS_FOR_MAIN_PY)

    print(f"\nRunning trial {trial.number} with command: {' '.join(command)}")

    # 3. اجرای main.py به عنوان یک subprocess
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=1800) # ۳۰ دقیقه مهلت زمانی
        if process.returncode != 0:
            print(f"Error in trial {trial.number}: Subprocess failed with return code {process.returncode}")
            print("STDERR:")
            print(process.stderr[-1000:]) # چاپ ۱۰۰۰ کاراکتر آخر خطا
            # اگر خطا رخ داد، این آزمایش را ناموفق در نظر بگیرید (با برگرداندن مقدار بد یا استثنا)
            # return -1 # یا مقداری که نشان دهنده شکست است
            raise optuna.exceptions.TrialPruned() # برای اینکه pruner آن را هرس کند

        # 4. استخراج متریک از خروجی (مثلا MRR@K روی مجموعه اعتبارسنجی)
        # شما باید این بخش را مطابق با فرمت خروجی main.py خود تنظیم کنید
        metric_value = parse_metric_from_output(process.stdout, metric_name="MRR") # یا "Recall"

        if metric_value is None:
            print(f"Warning: Could not parse metric for trial {trial.number}. Pruning.")
            raise optuna.exceptions.TrialPruned() # اگر متریک پیدا نشد، آزمایش را هرس کن

        # 5. گزارش متریک به Optuna (برای هرس)
        # `step` نشان دهنده میزان پیشرفت یا منبع مصرف شده است (مثلاً تعداد epoch)
        trial.report(metric_value, step=epochs_for_this_trial)

        # 6. بررسی اینکه آیا آزمایش باید هرس شود
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epochs_for_this_trial}.")
            raise optuna.exceptions.TrialPruned()

        return metric_value # Optuna سعی می کند این مقدار را بیشینه کند

    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out. Pruning.")
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"An unexpected error occurred in trial {trial.number}: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    # 1. انتخاب Pruner (هرس کننده)
    # SuccessiveHalvingPruner یکی از پیاده سازی های خانواده ASHA است
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=5,         # حداقل منبعی که به یک trial داده می شود (اینجا منظور تعداد epoch است)
        reduction_factor=2,     # در هر مرحله، چه کسری از trial ها نگه داشته می شوند (1/factor)
        min_early_stopping_rate=0 # تعداد مراحل اولیه که در آن ها هرس انجام نمی شود
    )
    # یا از MedianPruner برای سادگی بیشتر استفاده کنید:
    # pruner = optuna.pruners.MedianPruner(
    #     n_startup_trials=5,      # تعداد آزمایش های اولیه که برای محاسبه میانه استفاده می شوند
    #     n_warmup_steps=3,        # تعداد epoch های اولیه که قبل از شروع هرس صبر می شود
    #     interval_steps=1         # هر چند epoch یکبار بررسی برای هرس انجام شود
    # )

    # 2. ایجاد یک مطالعه (study) در Optuna
    # direction="maximize" چون می خواهیم MRR یا Recall را بیشینه کنیم
    study_name = f"session_rec_study_{DATASET_NAME}" # نام مطالعه برای ذخیره سازی احتمالی
    storage_name = f"sqlite:///{study_name}.db"      # ذخیره نتایج در یک دیتابیس SQLite

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name, # برای ادامه دادن مطالعه در صورت توقف
        load_if_exists=True,  # اگر مطالعه ای با این نام وجود دارد، آن را بارگذاری کن
        direction="maximize",
        pruner=pruner
    )

    # 3. اجرای بهینه سازی
    print(f"Starting Optuna study. Using {N_TRIALS_OPTUNA} trials.")
    print(f"Results will be saved in {storage_name}")
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA)

    # 4. چاپ بهترین نتایج
    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print("\nBest trial:")
    print(f"  Value (Maximized Metric e.g. MRR): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # می توانید بهترین پارامترها را برای اجرای نهایی روی کل دیتاست و با epoch بیشتر استفاده کنید
    print("\nTo run the best model manually (adjust epochs and remove subset ratio for final run):")
    best_args = []
    for key, value in best_trial.params.items():
        # تبدیل نام پارامترهای Optuna به آرگومان های خط فرمان
        # این بخش ممکن است نیاز به تنظیم دقیق داشته باشد
        if key == "epochs_for_trial": # برای اجرای نهایی، epoch را خودتان تنظیم کنید
            continue
        best_args.append(f"--{key}")
        best_args.append(str(value))

    final_command_parts = [
        PYTHON_EXECUTABLE, MAIN_SCRIPT_PATH,
        # "--epoch", "30", # تعداد epoch کامل برای اجرای نهایی
        # "--data_subset_ratio", "1.0", # استفاده از کل دیتا
    ]
    final_command_parts.extend(best_args)
    final_command_parts.extend(FIXED_ARGS_FOR_MAIN_PY)
    # حذف --data_subset_ratio و --epoch از fixed_args اگر در best_args آمده اند و با مقادیر نهایی جایگزینشان کنید
    # این بخش نیاز به مدیریت دقیق تر آرگومان ها دارد برای اجرای نهایی
    print(f"Approximate command for final run (customize --epoch and --data_subset_ratio):")
    print(f"{' '.join(final_command_parts).replace('--data_subset_ratio '+str(data_subset_for_this_trial), '--data_subset_ratio 1.0').replace('--epoch '+str(best_trial.params.get('epochs_for_trial','10')), '--epoch 30')}")