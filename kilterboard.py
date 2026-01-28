import os

# --- CRITICAL FIX FOR MAC ---
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
from sklearn.utils import resample
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. SETUP
# ==========================================
MAX_HOLDS = 30  # Reduced slightly to focus on signal
print("--- 1. LOADING MAP ---")
try:
    with open('kilter_layout.json', 'r') as f:
        layout = json.load(f)
    HOLD_DB = {i['placement_id']: {'x': i['x'], 'y': i['y']} for i in layout}
    print(f"✅ Loaded {len(HOLD_DB)} coordinates.")
except:
    print("❌ Error loading kilter_layout.json");
    exit()


# ==========================================
# 2. HYBRID FEATURE ENGINEERING
# ==========================================
def extract_hybrid_features(row):
    # 1. Parse Inputs
    input_text = row.get('instruction') or row.get('input')
    target_text = row.get('output') or row.get('target')

    diff_match = re.search(r'difficulty (\d+)', str(input_text))
    ang_match = re.search(r'angle (\d+)', str(input_text))
    if not diff_match: return None

    grade = int(diff_match.group(1))
    angle = int(ang_match.group(1)) if ang_match else 40

    # 2. Parse Holds
    matches = re.findall(r'p(\d+)r(\d+)', str(target_text))
    holds = []
    for pid, rid in matches:
        c = HOLD_DB.get(int(pid))
        if c: holds.append({'x': c['x'], 'y': c['y'], 'r': int(rid)})

    if len(holds) < 3: return None

    # 3. CALCULATE PHYSICS "HINTS" (The Aggregate Features)
    df_h = pd.DataFrame(holds)
    hands = df_h[df_h['r'].isin([12, 13, 14])]

    # Dimensions
    h_span = (hands['y'].max() - hands['y'].min()) if len(hands) > 0 else 0
    w_span = (hands['x'].max() - hands['x'].min()) if len(hands) > 0 else 0

    # Movement (Euclidean Distances)
    hands_sorted = hands.sort_values('y')
    if len(hands_sorted) > 1:
        dists = np.sqrt(np.diff(hands_sorted['x']) ** 2 + np.diff(hands_sorted['y']) ** 2)
        avg_move = np.mean(dists)
        max_move = np.max(dists)
        crux_factor = max_move / (avg_move + 1)  # How much harder is the hardest move?
    else:
        avg_move, max_move, crux_factor = 0, 0, 0

    # Density
    box_area = (h_span * w_span) + 1
    density = len(hands) / box_area

    # 4. SEQUENCE EMBEDDING (The "Nuance" Features)
    # Sort by height to align features
    holds.sort(key=lambda k: (k['y'], k['x']))

    # Vector Construction
    # [ --- PHYSICS HINTS --- , --- RAW SEQUENCE --- ]
    vector = [
        angle,
        len(hands),
        h_span,
        w_span,
        avg_move,
        max_move,
        crux_factor,
        density
    ]

    # Flatten sequence (X, Y, Role)
    for i in range(MAX_HOLDS):
        if i < len(holds):
            h = holds[i]
            # Normalize to ~0-1 range (Board is approx 150x150)
            vector.extend([h['x'] / 150, h['y'] / 150, h['r'] / 15])
        else:
            vector.extend([0, 0, 0])  # Padding

    return vector, grade, input_text


# ==========================================
# 3. PROCESSING & OVERSAMPLING
# ==========================================
print("\n--- 2. PROCESSING ---")
dataset = load_dataset("gabriead/Kilterboard", split="train")

X_list, y_list, seen = [], [], set()

for row in dataset:
    res = extract_hybrid_features(row)
    if res:
        vec, g, txt = res
        if txt not in seen:  # Dedup
            seen.add(txt)
            X_list.append(vec)
            y_list.append(g)

# Binning
y_binned = [0 if g < 14 else (1 if g < 22 else 2) for g in y_list]
df = pd.DataFrame(X_list)
df['target'] = y_binned

print(f"Unique Samples: {len(df)}")
print(f"Class Dist (Raw): {df['target'].value_counts().to_dict()}")

print("\n--- 3. OVERSAMPLING (Maximizing Data) ---")
# Instead of throwing away data (undersampling), we duplicate the small classes
max_size = df['target'].value_counts().max()

df_balanced = pd.concat([
    resample(df[df['target'] == c], replace=True, n_samples=max_size, random_state=42)
    for c in df['target'].unique()
])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Training Size: {len(df_balanced)} ({max_size} per class)")

X = df_balanced.drop('target', axis=1).values
y = df_balanced['target'].values

# ==========================================
# 4. BENCHMARK
# ==========================================
print("\n--- 4. RUNNING BENCHMARK ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pfn_scores, xgb_scores = [], []

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # TabPFN (Increased ensemble for better complex feature handling)
    # Using N=8 to keep speed reasonable, N=32 is best
    pfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
    pfn.fit(X_train, y_train)
    pfn_acc = accuracy_score(y_test, pfn.predict(X_test))
    pfn_scores.append(pfn_acc)

    # XGBoost
    xgb = XGBClassifier(n_jobs=1, objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    xgb_scores.append(xgb_acc)

    print(f"Fold {i + 1}: TabPFN={pfn_acc:.2%} | XGBoost={xgb_acc:.2%}")

print("-" * 40)
print(f"TabPFN Avg: {np.mean(pfn_scores):.2%} (+/- {np.std(pfn_scores):.2%})")
print(f"XGBoost Avg: {np.mean(xgb_scores):.2%} (+/- {np.std(xgb_scores):.2%})")
print("-" * 40)