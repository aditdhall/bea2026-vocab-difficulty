"""
OVERNIGHT BATCH — Run All Remaining Training Experiments
=========================================================
Paste this into a single Colab cell and run before bed.
Estimated time: ~7-8 hours on A100.

Order (fastest to slowest, dependencies respected):
1. Exp 0.5 — Target scaling (~30 min)
2. Exp 3  — Structured input (~30 min)
3. Exp 2  — mdeberta comparison (~1 hr)
4. Exp 5  — Hyperparameter tuning (~2 hrs)
5. Ablation table (~2.8 hrs)
6. SHAP on final model (~15 min)

All results saved to results/ and copied to Drive.
"""
import time
start_time = time.time()

def elapsed():
    mins = (time.time() - start_time) / 60
    return f"[{mins:.0f} min elapsed]"

# ============================================================
# 1. EXP 0.5 — Target Scaling
# ============================================================
print(f"\n{'#'*70}")
print(f"  1/6: EXP 0.5 — Target Scaling {elapsed()}")
print(f"{'#'*70}")
%run scripts/colab_04_exp05.py

# ============================================================
# 2. EXP 3 — Structured Input
# ============================================================
print(f"\n{'#'*70}")
print(f"  2/6: EXP 3 — Structured Input {elapsed()}")
print(f"{'#'*70}")
%run scripts/colab_04_exp3.py

# ============================================================
# 3. EXP 2 — mdeberta comparison
# ============================================================
print(f"\n{'#'*70}")
print(f"  3/6: EXP 2 — mdeberta-v3-base {elapsed()}")
print(f"{'#'*70}")
%run scripts/colab_04_exp2.py

# ============================================================
# 4. EXP 5 — Hyperparameter Tuning
# ============================================================
print(f"\n{'#'*70}")
print(f"  4/6: EXP 5 — Hyperparameter Tuning {elapsed()}")
print(f"{'#'*70}")
%run scripts/colab_04_exp5.py

# ============================================================
# 5. ABLATION TABLE
# ============================================================
print(f"\n{'#'*70}")
print(f"  5/6: ABLATION TABLE {elapsed()}")
print(f"{'#'*70}")
%run scripts/colab_05_ablation.py

# ============================================================
# 6. SHAP ON FINAL MODEL
# ============================================================
print(f"\n{'#'*70}")
print(f"  6/6: SHAP ON FINAL MODEL {elapsed()}")
print(f"{'#'*70}")
%run scripts/colab_05_shap.py

# ============================================================
# SAVE EVERYTHING TO DRIVE
# ============================================================
print(f"\n{'#'*70}")
print(f"  SAVING ALL RESULTS TO DRIVE {elapsed()}")
print(f"{'#'*70}")

import subprocess
subprocess.run(['mkdir', '-p', '/content/drive/MyDrive/bea2026/results/predictions'])
subprocess.run(['mkdir', '-p', '/content/drive/MyDrive/bea2026/results/figures'])

import glob
for f in glob.glob('results/predictions/*.csv'):
    subprocess.run(['cp', f, '/content/drive/MyDrive/bea2026/results/predictions/'])
for f in glob.glob('results/figures/*.pdf'):
    subprocess.run(['cp', f, '/content/drive/MyDrive/bea2026/results/figures/'])
for f in glob.glob('results/*.csv'):
    subprocess.run(['cp', f, '/content/drive/MyDrive/bea2026/results/'])

total_mins = (time.time() - start_time) / 60
print(f"\n{'='*70}")
print(f"  ALL DONE! Total time: {total_mins:.0f} minutes ({total_mins/60:.1f} hours)")
print(f"{'='*70}")
print(f"\nResults saved to Google Drive.")
print(f"Check: /content/drive/MyDrive/bea2026/results/")
