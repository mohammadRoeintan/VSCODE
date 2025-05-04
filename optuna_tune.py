import optuna
import subprocess
import sys

def objective(trial):
    hidden_size = trial.suggest_categorical('hiddenSize', [50, 100, 150, 200])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    l2 = trial.suggest_float('l2', 1e-6, 1e-4, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

    # دستور اجرای فایل main.py با پارامترهای پیشنهادی
    cmd = [
        sys.executable, 'main.py',
        '--dataset', 'yoochoose1_64',
        '--hiddenSize', str(hidden_size),
        '--lr', str(lr),
        '--l2', str(l2),
        '--dropout', str(dropout),
        '--epoch', '10',  # سریع برای tuning
        '--patience', '5'  # سریع برای tuning
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        # آخرین Recall@20 را از لاگ استخراج کن
        last_line = [line for line in output.split('\n') if 'Best Result' in line]
        if last_line:
            recall_lines = [line for line in result.stdout.splitlines() if 'Recall@20:' in line]
            if not recall_lines:
                print("No Recall@20 line found in output!")
                print("Full output:\n", result.stdout)
                raise RuntimeError("Failed to extract Recall@20 from output.")
            recall_line = recall_lines[0]

            recall_value_str = recall_line.split('Recall@20:')[1].split('\t')[0].strip()
            if not recall_value_str:
                print("Recall@20 value string is empty!")
                print("Recall line:\n", recall_line)
                raise RuntimeError("Recall@20 value is empty.")
            recall_value = float(recall_value_str)


            return recall_value
        else:
            return 0.0  # اگر پیدا نشد
    except subprocess.CalledProcessError as e:
        print("Error running main.py:", e)
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return 0.0

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # تعداد تست‌ها: ۲۰

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
