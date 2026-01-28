import os
# --- CRITICAL FIXES FOR MAC ---
# 1. Disable OpenMP threading to prevent XGBoost/PyTorch conflict
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# 2. Allow duplicate libraries (common fix for XGBoost on Mac)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from datasets import load_dataset
import warnings
import torch

warnings.filterwarnings('ignore')

# --- 3. FORCE CPU FOR STABILITY ---
# While MPS (GPU) is faster, it causes crashes when mixed with XGBoost on some M1/M2 chips.
# For N=160 samples, CPU is fast enough (approx 1-2 seconds).
device = 'cpu'

print(f"--- 1. LOADING DATA (Device: {device}) ---")

dataset = load_dataset("marianeft/diabetes_prediction_dataset", split="train")
df = dataset.to_pandas()

# Simple Encoding
gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
smoke_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
df['gender'] = df['gender'].map(gender_map)
df['smoking_history'] = df['smoking_history'].map(smoke_map)

X = df[['age', 'gender', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'smoking_history']]
y = df['diabetes']

# Hold out large test set
X_unused, X_test, y_unused, y_test = train_test_split(X, y, test_size=2000, random_state=42, stratify=y)

# ==========================================
# 2. RUNNING THE BENCHMARK
# ==========================================
sample_sizes = [40, 80, 160]

print(f"\n--- 2. RUNNING 'RARE DISEASE' BENCHMARK ---")
print(f"{'Samples (N)':<15} | {'TabPFN AUC':<15} | {'XGBoost AUC':<15} | {'Winner':<10}")
print("-" * 65)

for n_train in sample_sizes:
    # Create tiny training set
    X_train_rare, _, y_train_rare, _ = train_test_split(
        X_unused, y_unused, train_size=n_train, random_state=42, stratify=y_unused
    )

    # --- MODEL A: TABPFN ---
    # N_ensemble_configurations=8 is enough for a quick demo
    pfn = TabPFNClassifier(device=device)
    pfn.fit(X_train_rare, y_train_rare)
    pfn_pred = pfn.predict_proba(X_test)[:, 1]
    pfn_auc = roc_auc_score(y_test, pfn_pred)

    # --- MODEL B: XGBOOST ---
    # n_jobs=1 ensures it respects the environment variables we set above
    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, n_jobs=1)
    xgb.fit(X_train_rare, y_train_rare)
    xgb_pred = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred)

    winner = "TabPFN" if pfn_auc > xgb_auc else "XGBoost"
    print(f"{n_train:<15} | {pfn_auc:.4f}          | {xgb_auc:.4f}          | {winner}")

print("-" * 65)