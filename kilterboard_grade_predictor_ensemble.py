import os
import sys

# --- SAFETY CONFIG ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.neighbors import NearestNeighbors

# Force Single Thread
torch.set_num_threads(1)

print("✅ Environment Configured for Ensemble Comparison.")

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
MAX_HOLDS = 30
TABPFN_CONTEXT_SIZE = 2048
INFERENCE_BATCH_SIZE = 32  # Safe for MPS
INFERENCE_SAMPLE_SIZE = 2000

# ==========================================
# 2. DATA LOADING (STRICT QUALITY)
# ==========================================
print("\n--- 1. LOADING DATA ---")

try:
    with open('kilter_layout_orig.json', 'r') as f:
        layout = json.load(f)
    HOLD_DB = {i['placement_id']: {'x': i['x'], 'y': i['y']} for i in layout}
    print(f"   Loaded {len(HOLD_DB)} hold coordinates.")
except FileNotFoundError:
    print("❌ Error: 'kilter_layout_orig.json' not found.")
    sys.exit()

try:
    print("   Loading climbs definitions...")
    with open('climbs.json', 'r') as f:
        climbs_raw = json.load(f)
    uuid_to_frames = {item['uuid']: item['frames'] for item in climbs_raw}
    print(f"   Mapped {len(uuid_to_frames)} climb definitions.")
except FileNotFoundError:
    print("❌ Error: 'climbs.json' not found.")
    sys.exit()

try:
    print("   Loading stats & applying Golden Filter...")
    df_stats = pd.read_json('climb_stats.json')

    # --- THE FILTER (Exact match to your request) ---
    condition = (
            (df_stats['display_difficulty'] == df_stats['difficulty_average']) &
            (df_stats['display_difficulty'] % 1 == 0) &
            (df_stats['quality_average'] == 3)
    )

    valid_stats = df_stats[condition].copy()
    print(f"   Found {len(valid_stats)} Golden Archetypes (stats match).")

except FileNotFoundError:
    print("❌ Error: 'climb_stats.json' not found.")
    sys.exit()


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def get_bin(grade):
    # Map raw difficulty (e.g., 10-33) to bins (0-7)
    # Ignore <10 to prevent negative class indices
    if grade < 10: return None
    if grade >= 33: return None
    return (grade - 10) // 3


def extract_features(frames_str, angle, grade):
    if not frames_str: return None

    grade_bin = get_bin(grade)
    if grade_bin is None: return None

    matches = re.findall(r'p(\d+)r(\d+)', frames_str)

    holds = []
    for p, r in matches:
        pid = int(p)
        if pid in HOLD_DB:
            holds.append({
                'x': HOLD_DB[pid]['x'],
                'y': HOLD_DB[pid]['y'],
                'r': int(r)
            })

    if len(holds) < 3: return None

    # Physics Calculations
    df_h = pd.DataFrame(holds)
    hands = df_h[df_h['r'].isin([12, 13, 14, 15])]

    if len(hands) > 0:
        h_span = float(hands['y'].max() - hands['y'].min())
        w_span = float(hands['x'].max() - hands['x'].min())
        density = len(hands) / ((h_span * w_span) + 1.0)
    else:
        h_span, w_span, density = 0.0, 0.0, 0.0

    if len(hands) > 1:
        hands_sort = hands.sort_values('y')
        dists = np.sqrt(np.diff(hands_sort['x']) ** 2 + np.diff(hands_sort['y']) ** 2)
        avg_move = float(np.mean(dists))
        max_move = float(np.max(dists))
        crux_ratio = max_move / (avg_move + 1.0)
    else:
        avg_move, max_move, crux_ratio = 0.0, 0.0, 0.0

    holds.sort(key=lambda k: (k['y'], k['x']))

    vec = [
        float(angle) / 70.0,
        float(len(hands)) / 20.0,
        h_span / 150.0,
        w_span / 150.0,
        max_move / 50.0,
        crux_ratio / 5.0,
        density * 10.0
    ]

    for i in range(MAX_HOLDS):
        if i < len(holds):
            vec.extend([
                float(holds[i]['x']) / 150.0,
                float(holds[i]['y']) / 150.0,
                float(holds[i]['r']) / 15.0
            ])
        else:
            vec.extend([0.0, 0.0, 0.0])

    return np.array(vec, dtype=np.float32), grade_bin


# ==========================================
# 4. BUILDING DATASET
# ==========================================
print("\n--- 2. BUILDING DATASET ---")
data_X, data_y, data_angles = [], [], []

for _, row in tqdm(valid_stats.iterrows(), total=len(valid_stats)):
    uuid = row['climb_uuid']
    if uuid in uuid_to_frames:
        res = extract_features(uuid_to_frames[uuid], int(row['angle']), int(row['display_difficulty']))
        if res:
            data_X.append(res[0])
            data_y.append(res[1])
            data_angles.append(int(row['angle']))

X = np.stack(data_X)
y = np.array(data_y)
angles = np.array(data_angles)

print(f"   Total Valid Samples: {len(X)}")

# Split into Train and Test
X_train_raw, X_test, y_train_raw, y_test, angles_train, angles_test = train_test_split(
    X, y, angles, test_size=0.2, random_state=42
)

# ==========================================
# 5. PREPARING EXPERT CONTEXTS
# ==========================================
print("\n--- 3. CREATING EXPERTS (Vertical, Overhang, Steep) ---")


def get_expert_context(min_angle, max_angle, max_samples=2048):
    """
    Selects balanced training samples specifically for a given angle range.
    """
    # 1. Filter by Angle
    mask = (angles_train >= min_angle) & (angles_train <= max_angle)
    indices = np.where(mask)[0]

    if len(indices) == 0: return None, None

    # 2. Stratify Selection (Archetyping)
    X_pool = X_train_raw[indices]
    y_pool = y_train_raw[indices]

    # Create DataFrame to group by Grade
    df_pool = pd.DataFrame({'y': y_pool, 'original_idx': indices})

    selected_indices = []
    # Calculate how many per grade we can take
    unique_grades = df_pool['y'].unique()
    per_grade = max(1, max_samples // len(unique_grades))

    for g in unique_grades:
        subset = df_pool[df_pool['y'] == g]

        # 3. Use Centroid method to find "Representative" climbs for this grade/angle
        X_sub = X_train_raw[subset['original_idx'].values]
        centroid = np.mean(X_sub, axis=0).reshape(1, -1)

        take = min(len(X_sub), per_grade)
        nbrs = NearestNeighbors(n_neighbors=take).fit(X_sub)
        _, local_idxs = nbrs.kneighbors(centroid)

        # Add to selection
        selected_indices.extend(subset['original_idx'].values[local_idxs[0]])

    # Shuffle
    np.random.shuffle(selected_indices)

    # Trim to Max
    selected_indices = selected_indices[:max_samples]

    # Sanitize for MPS
    X_ctx = np.ascontiguousarray(X_train_raw[selected_indices], dtype=np.float32)
    y_ctx = np.ascontiguousarray(y_train_raw[selected_indices])

    return X_ctx, y_ctx


# --- Generate Contexts ---
# Expert 1: Vertical (0-35 deg)
X_c1, y_c1 = get_expert_context(0, 35)
print(f"   Expert 1 (Vertical): {X_c1.shape}")

# Expert 2: Overhang (40-50 deg)
X_c2, y_c2 = get_expert_context(40, 50)
print(f"   Expert 2 (Overhang): {X_c2.shape}")

# Expert 3: Steep (55-70 deg)
X_c3, y_c3 = get_expert_context(55, 70)
print(f"   Expert 3 (Steep):    {X_c3.shape}")

# ==========================================
# 6. TRAINING (6 Models Total)
# ==========================================
print("\n--- 4. TRAINING ENSEMBLES ---")

# --- XGBoost Ensemble ---
print("Training 3 XGBoost Experts...")
xgb_vert = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=1)
xgb_over = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=1)
xgb_steep = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=1)

xgb_vert.fit(X_c1, y_c1)
xgb_over.fit(X_c2, y_c2)
xgb_steep.fit(X_c3, y_c3)

# --- TabPFN Ensemble ---
print("Initializing 3 TabPFN Experts...")
pfn_vert = TabPFNClassifier(device='mps', n_estimators=1, n_jobs=4, ignore_pretraining_limits=True)
pfn_over = TabPFNClassifier(device='mps', n_estimators=1, n_jobs=4, ignore_pretraining_limits=True)
pfn_steep = TabPFNClassifier(device='mps', n_estimators=1, n_jobs=4, ignore_pretraining_limits=True)

pfn_vert.fit(X_c1, y_c1)
pfn_over.fit(X_c2, y_c2)
pfn_steep.fit(X_c3, y_c3)

# ==========================================
# 7. INFERENCE (Ensemble Voting)
# ==========================================
print(f"\n--- 5. INFERENCE (Sampling {INFERENCE_SAMPLE_SIZE}) ---")

# Sample Test Set
num_samples = min(INFERENCE_SAMPLE_SIZE, len(X_test))
sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

X_test_sample = np.ascontiguousarray(X_test[sample_indices])
y_test_sample = np.ascontiguousarray(y_test[sample_indices])

# --- XGBoost Inference ---
print("Predicting XGBoost Ensemble...")
# We get probabilities from all 3 and average them (Soft Voting)
xgb_p1 = xgb_vert.predict_proba(X_test_sample)
xgb_p2 = xgb_over.predict_proba(X_test_sample)
xgb_p3 = xgb_steep.predict_proba(X_test_sample)


# Handle cases where experts might see fewer classes than total (0-7)
# (Helper to pad probability arrays if an expert missed a grade class)
def pad_probas(proba, n_classes=8):
    if proba.shape[1] == n_classes: return proba
    # If missing columns, simplistic approach: assume 0 prob for missing
    # (In strictly filtered data this shouldn't happen often if stratification worked)
    padded = np.zeros((len(proba), n_classes))
    # This assumes classes are 0..N, might need smarter mapping if classes are skipped
    # For now, simplistic padding to prevent crash
    padded[:, :proba.shape[1]] = proba
    return padded


xgb_p1 = pad_probas(xgb_p1)
xgb_p2 = pad_probas(xgb_p2)
xgb_p3 = pad_probas(xgb_p3)

xgb_final_prob = (xgb_p1 + xgb_p2 + xgb_p3) / 3.0
xgb_preds = np.argmax(xgb_final_prob, axis=1)

# --- TabPFN Inference ---
print("Predicting TabPFN Ensemble...")
pfn_probs_acc = np.zeros((len(y_test_sample), 8))

# Batched Inference
for i in tqdm(range(0, len(X_test_sample), INFERENCE_BATCH_SIZE)):
    batch = X_test_sample[i: i + INFERENCE_BATCH_SIZE]

    # Get probs from all 3 experts
    # Note: TabPFN handles class padding internally better usually
    pp1 = pfn_vert.predict_proba(batch)
    pp2 = pfn_over.predict_proba(batch)
    pp3 = pfn_steep.predict_proba(batch)

    pp1 = pad_probas(pp1)
    pp2 = pad_probas(pp2)
    pp3 = pad_probas(pp3)

    avg_batch = (pp1 + pp2 + pp3) / 3.0

    # Accumulate
    pfn_probs_acc[i: i + len(batch)] = avg_batch

pfn_preds = np.argmax(pfn_probs_acc, axis=1)


# ==========================================
# 8. EVALUATION
# ==========================================
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pm1 = np.mean(np.abs(y_pred - y_true) <= 1)
    return acc, pm1


x_acc, x_pm1 = evaluate(y_test_sample, xgb_preds)
t_acc, t_pm1 = evaluate(y_test_sample, pfn_preds)

print("\n" + "=" * 60)
print(f"{'METRIC':<25} | {'XGB ENSEMBLE':<12} | {'PFN ENSEMBLE':<12}")
print("=" * 60)
print(f"{'Exact Accuracy':<25} | {x_acc:.1%}       | {t_acc:.1%}")
print(f"{'+/- 1 Bin':<25} | {x_pm1:.1%}       | {t_pm1:.1%}")
print("=" * 60)

# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test_sample, xgb_preds), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"XGBoost Ensemble\nAcc: {x_acc:.1%}")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test_sample, pfn_preds), annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title(f"TabPFN Ensemble\nAcc: {t_acc:.1%}")

plt.tight_layout()
plt.savefig('ensemble_comparison_results.png')
print("✅ Saved results to ensemble_comparison_results.png")