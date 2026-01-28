import os

# --- CRITICAL FIX FOR MAC (M1/M2/M3) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import json
import re
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. LOAD THE "GOLDEN KEY" (COORDINATE MAP)
# ==========================================
print("--- 1. LOADING COORDINATE MAP ---")

try:
    with open('kilter_layout.json', 'r') as f:
        layout_list = json.load(f)
    HOLD_DB = {item['placement_id']: {'x': item['x'], 'y': item['y']} for item in layout_list}
    print(f"✅ Successfully loaded real coordinates for {len(HOLD_DB)} holds.")
except FileNotFoundError:
    print("❌ ERROR: 'kilter_layout.json' not found.")
    exit()
except json.JSONDecodeError:
    print("❌ ERROR: 'kilter_layout.json' contains invalid JSON.")
    exit()


# ==========================================
# 2. PARSER
# ==========================================
def parse_instruction(text):
    if not isinstance(text, str): return {'difficulty': None, 'angle': None}
    diff = re.search(r'difficulty (\d+)', text)
    ang = re.search(r'angle (\d+)', text)
    return {
        'difficulty': int(diff.group(1)) if diff else None,
        'angle': int(ang.group(1)) if ang else 40
    }


def parse_route(route_str):
    if not isinstance(route_str, str): return []
    matches = re.findall(r'p(\d+)r(\d+)', route_str)
    holds = []
    for pid, rid in matches:
        coords = HOLD_DB.get(int(pid))
        if coords:
            holds.append({'id': int(pid), 'role': int(rid), 'x': coords['x'], 'y': coords['y']})
    return holds


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def extract_features(row):
    # Handle column name variations (HuggingFace vs Custom)
    input_text = row.get('instruction') or row.get('input')
    target_text = row.get('output') or row.get('target')

    meta = parse_instruction(input_text)
    target_grade = meta['difficulty']
    holds = parse_route(target_text)

    if not holds or target_grade is None: return None

    df_h = pd.DataFrame(holds)
    hands = df_h[df_h['role'].isin([12, 13, 14])]
    feet = df_h[df_h['role'] == 15]

    if len(hands) < 2: return None

    # Physics Features
    height = hands['y'].max() - hands['y'].min()
    width = hands['x'].max() - hands['x'].min()

    hands_sorted = hands.sort_values('y')
    dists = np.sqrt(np.diff(hands_sorted['x']) ** 2 + np.diff(hands_sorted['y']) ** 2)
    avg_move_dist = np.mean(dists) if len(dists) > 0 else 0
    max_move_dist = np.max(dists) if len(dists) > 0 else 0

    area = (height * width) + 1
    density = len(hands) / area
    center_y = hands['y'].mean()

    return [
        meta['angle'], len(hands), len(feet), width, height,
        avg_move_dist, max_move_dist, density, center_y
    ], target_grade


# ==========================================
# 4. DATA PREP & BALANCING
# ==========================================
print("\n--- 2. PROCESSING DATASET ---")
dataset = load_dataset("gabriead/Kilterboard", split="train")

X_raw, y_raw = [], []
for i, row in enumerate(dataset):
    if i > 5000: break  # Process more rows to ensure we have enough for balancing
    result = extract_features(row)
    if result:
        X_raw.append(result[0])
        y_raw.append(result[1])

# Create DataFrame for easy manipulation
feature_names = ['Angle', 'N_Hands', 'N_Feet', 'Width', 'Height', 'Avg_Dist', 'Max_Dist', 'Density', 'Center_Y']
df_full = pd.DataFrame(X_raw, columns=feature_names)
# Bin grades: 0=Easy (<14), 1=Medium (14-22), 2=Hard (>22)
df_full['grade_bin'] = [0 if g < 14 else (1 if g < 22 else 2) for g in y_raw]

print(f"Total samples extracted: {len(df_full)}")
print(f"Class distribution before balancing:\n{df_full['grade_bin'].value_counts()}")

# --- BALANCING (UNDERSAMPLING) ---
print("\n--- BALANCING & SHUFFLING ---")
min_class_size = df_full['grade_bin'].value_counts().min()
df_balanced = pd.concat([
    df_full[df_full['grade_bin'] == label].sample(min_class_size, random_state=42)
    for label in df_full['grade_bin'].unique()
])

# --- SHUFFLING ---
df_balanced = shuffle(df_balanced, random_state=42).reset_index(drop=True)

X = df_balanced[feature_names].values
y = df_balanced['grade_bin'].values

print(f"Balanced dataset size: {len(df_balanced)} ({min_class_size} per class)")

# ==========================================
# 5. STRATIFIED K-FOLD CROSS-VALIDATION
# ==========================================
print("\n--- 3. CROSS-VALIDATION BENCHMARK ---")

# 5-Fold Stratified Split ensures every fold has equal class proportions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pfn_scores = []
xgb_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- TabPFN ---
    # N_ensemble_configurations=4 is fast; use 32 for max accuracy in production
    pfn = TabPFNClassifier(device='cpu')
    pfn.fit(X_train, y_train)
    pfn_pred = pfn.predict(X_test)
    pfn_acc = accuracy_score(y_test, pfn_pred)
    pfn_scores.append(pfn_acc)

    # --- XGBoost ---
    # Fixed parameters for Multi-Class classification
    xgb = XGBClassifier(
        n_jobs=1,  # Fix Mac crash
        objective='multi:softmax',  # Fix logistic loss error
        num_class=3,  # 3 Classes (Easy, Med, Hard)
        eval_metric='mlogloss',  # Multi-class log loss
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_scores.append(xgb_acc)

    print(f"Fold {fold + 1}: TabPFN={pfn_acc:.2%} | XGBoost={xgb_acc:.2%}")

# ==========================================
# 6. FINAL RESULTS
# ==========================================
print("-" * 40)
print(f"AVERAGE ACCURACY (5-Fold CV)")
print(f"TabPFN:  {np.mean(pfn_scores):.2%} (+/- {np.std(pfn_scores):.2%})")
print(f"XGBoost: {np.mean(xgb_scores):.2%} (+/- {np.std(xgb_scores):.2%})")
print("-" * 40)