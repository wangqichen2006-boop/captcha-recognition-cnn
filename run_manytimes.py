"""
Runs main_train.py multiple times in a row.

Usage:
    python3 run_multiple_times.py 10
    (runs training 10 times back-to-back)

If no number is given, defaults to 5 runs.
"""

import sys
import subprocess
import time

TRAIN_SCRIPT = "train_digits.py"   # change this if your training script has a different name

# ---- Parse how many times to run ----
if len(sys.argv) > 1:
    try:
        num_runs = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid number, e.g.: python3 run_multiple_times.py 10")
        sys.exit(1)
else:
    num_runs = 5
    print(f"No number given, defaulting to {num_runs} runs.")

print(f">>> Will run '{TRAIN_SCRIPT}' {num_runs} time(s) in a row.\n")

start_time = time.time()

for i in range(1, num_runs + 1):
    print(f"\n{'='*50}")
    print(f"  STARTING TRAINING PASS {i}/{num_runs}")
    print(f"{'='*50}\n")

    result = subprocess.run(["python3", TRAIN_SCRIPT])

    if result.returncode != 0:
        print(f"\n!!! Training pass {i} failed (exit code {result.returncode}). Stopping.")
        sys.exit(1)

elapsed = time.time() - start_time
print(f"\n>>> All {num_runs} training passes completed successfully.")
print(f">>> Total time: {elapsed/60:.1f} minutes")
print(">>> Check accuracy_history.png to see the accuracy trend across all runs.")