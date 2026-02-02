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
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.manifold import TSNE
import shap

# CRITICAL: Force CPU to avoid MPS "Silent Corruption"
torch.set_num_threads(4)

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
# 10 Global + 6 Dense Statistics = 16 Features Total
EXPECTED_VECTOR_LEN = 14

TABPFN_CONTEXT_SIZE = 2048
INFERENCE_BATCH_SIZE = 128
INFERENCE_SAMPLE_SIZE = 2000

# ==========================================
# 2. DATA LOADING (Robust)
# ==========================================
print("\n--- 1. LOADING DATA ---")


def load_json_anywhere(filename):
    candidates = [filename, f"data/{filename}"]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r') as f: return json.load(f)
    print(f"❌ Error: Could not find '{filename}'")
    sys.exit()


# 1. Layout
try:
    layout = load_json_anywhere('kilter_layout.json')
    HOLD_DB = {i['placement_id']: {'x': i['x'], 'y': i['y']} for i in layout}
except:
    layout = load_json_anywhere('kilter_layout_orig.json')
    HOLD_DB = {i['placement_id']: {'x': i['x'], 'y': i['y']} for i in layout}

# 2. Climbs
climbs_raw = load_json_anywhere('climbs.json')
uuid_to_data = {item['uuid']: item for item in climbs_raw if item.get("layout_id") == 1}
print(f"   Loaded {len(uuid_to_data)} Climbs (Layout 1).")

# 3. Stats
if os.path.exists('climb_stats.json'):
    df_stats = pd.read_json('climb_stats.json')
elif os.path.exists('data/climb_stats.json'):
    df_stats = pd.read_json('data/climb_stats.json')
else:
    print("❌ Error: climb_stats.json not found")
    sys.exit()

condition = (
        (df_stats['display_difficulty'] == df_stats['difficulty_average']) &
        (df_stats['display_difficulty'] % 1 == 0) &
        (df_stats['quality_average'] == 3) &
        (df_stats['angle'].notna())
)
valid_stats = df_stats[condition].copy()
print(f"   Found {len(valid_stats)} Golden Archetypes.")

# ==========================================
# 3. FEATURE ENGINEERING (DENSE)
# ==========================================
feature_names = [
    "Angle", "Hand Count", "Height", "Aspect Ratio",
    "Density", "Crux Ratio", "Start Height", "Finish Height",
    "Mean X", "Std X", "Mean Y", "Std Y", "Hand Count Stats", "Foot Count Stats"
]


def get_bin(grade):
    # Precise V-Scale Mapping (Capped at 10 Classes for TabPFN)
    if 10 <= grade <= 12: return 0  # V0
    if 13 <= grade <= 14: return 1  # V1
    if grade == 15: return 2  # V2
    if 16 <= grade <= 17: return 3  # V3
    if 18 <= grade <= 19: return 4  # V4
    if 20 <= grade <= 21: return 5  # V5
    if grade == 22: return 6  # V6
    if grade == 23: return 7  # V7
    if 24 <= grade <= 25: return 8  # V8
    # MERGE V9+ into Class 9 to stay within TabPFN limit (10 classes max)
    if grade >= 26: return 9  # V9+ (Elite)
    return None


def extract_features(climb_data, angle, grade):
    frames_str = climb_data.get('frames', '')
    if not frames_str: return None

    grade_bin = get_bin(grade)
    if grade_bin is None: return None

    # Parse Holds
    matches = re.findall(r'p(\d+)r(\d+)', frames_str)
    temp_holds = []
    for p, r in matches:
        pid = int(p)
        if pid in HOLD_DB:
            temp_holds.append({
                'x': HOLD_DB[pid]['x'],
                'y': HOLD_DB[pid]['y'],
                'r': int(r)
            })

    if len(temp_holds) < 3: return None

    # --- 1. GLOBAL GEOMETRY ---
    e_left = float(climb_data.get('edge_left', 0))
    e_right = float(climb_data.get('edge_right', 0))
    e_bot = float(climb_data.get('edge_bottom', 0))
    e_top = float(climb_data.get('edge_top', 0))
    width = e_right - e_left
    height = e_top - e_bot
    area = (width * height) + 1.0
    aspect_ratio = width / (height + 1.0)
    density = len(temp_holds) / area

    # --- 2. DENSE STATISTICS ---
    xs = [h['x'] for h in temp_holds]
    ys = [h['y'] for h in temp_holds]
    roles = [h['r'] for h in temp_holds]

    mean_x = np.mean(xs)
    std_x = np.std(xs)  # Horizontal spread
    mean_y = np.mean(ys)
    std_y = np.std(ys)  # Vertical spread

    # Count hold types (12=Start, 13=Hand, 14=Finish, 15=Foot)
    n_hand = roles.count(13)
    n_foot = roles.count(15)

    # --- 3. PHYSICS HINTS ---
    hands = [h for h in temp_holds if h['r'] in (12, 13, 14, 15)]
    if len(hands) > 1:
        hands.sort(key=lambda h: h['y'])
        h_xs = np.array([h['x'] for h in hands])
        h_ys = np.array([h['y'] for h in hands])
        dists = np.sqrt(np.diff(h_xs) ** 2 + np.diff(h_ys) ** 2)
        max_move = float(np.max(dists)) if len(dists) > 0 else 0.0
        avg_move = float(np.mean(dists)) if len(dists) > 0 else 0.0
        crux_ratio = max_move / (avg_move + 1.0)
    else:
        max_move, crux_ratio = 0.0, 0.0

    # --- 4. VECTOR CONSTRUCTION ---
    vec = [
        # Global (10)
        float(angle) / 70.0,
        float(len(hands)) / 20.0,
        # width / 144.0,
        height / 156.0,
        aspect_ratio,
        density * 1000.0,
        # max_move / 50.0,
        crux_ratio / 5.0,
        e_bot / 156.0,
        e_top / 156.0,
        # Dense Stats (6)
        mean_x / 144.0,
        std_x / 50.0,
        mean_y / 156.0,
        std_y / 50.0,
        n_hand / 20.0,
        n_foot / 20.0
    ]

    if len(vec) != EXPECTED_VECTOR_LEN: return None

    return np.array(vec, dtype=np.float32), grade_bin


# ==========================================
# 4. DATASET BUILD & BALANCING
# ==========================================
print("\n--- 2. BUILDING DATASET ---")
data_X, data_y = [], []
for _, row in tqdm(valid_stats.iterrows(), total=len(valid_stats)):
    uuid = row['climb_uuid']
    if uuid in uuid_to_data:
        res = extract_features(uuid_to_data[uuid], int(row['angle']), int(row['display_difficulty']))
        if res:
            vector, label = res
            if np.any(vector) and not np.isnan(vector).any():
                data_X.append(vector)
                data_y.append(res[1])

if not data_X:
    print("❌ Error: No data generated.")
    sys.exit()

X_raw = np.stack(data_X)
y_raw = np.array(data_y)

# --- CLASS BALANCING LOGIC ---
print("\n⚖️  BALANCING CLASSES...")
counts = Counter(y_raw)
print(f"   Original Counts: {dict(sorted(counts.items()))}")
min_count = min(counts.values())
print(f"   Target Samples per Class: {min_count}")

balanced_indices = []
indices = []

for label in counts.keys():
    indices = np.where(y_raw == label)[0]
    selected = np.random.choice(indices, min_count, replace=False)
    balanced_indices.extend(selected)

X = X_raw[balanced_indices]
y = y_raw[balanced_indices]

print(f"   Balanced Dataset Size: {len(X)} ({min_count} per class)")
print(f"   Features: {X.shape[1]}")

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 5. CONTEXT & CLUSTERING
# ==========================================
print("\n--- 3. CONTEXT & CLUSTERING ---")


def plot_grades():
    sample_size = min(len(X), 1000)
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_emb = tsne.fit_transform(X_sub)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y_sub, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Grade Bin')
    plt.title(f"t-SNE of Dense Features (N={sample_size})")
    plt.savefig("cluster.png")
    print("✅ Saved cluster.png")


# --- t-SNE Visualization ---
plot_grades()

# ==========================================
# 6. TRAINING XGBoost
# ==========================================
print("\n--- 4. TRAINING ---")
print("Fitting XGBoost...")

X_context = np.ascontiguousarray(X_train_raw, dtype=np.float32)
y_context = np.ascontiguousarray(y_train_raw)
print(f"   Context Set: {X_context.shape}")

xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=4)
xgb.fit(X_context, y_context)


def plot_xgb_feature_importance(feature_names, indices):
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("XGBoost Dense Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('xgb_feature_importance.png')
    print("✅ Saved XGB importance.")


plot_xgb_feature_importance(feature_names, indices)

# ==========================================
# 7. INFERENCE & COMPARISON
# ==========================================
is_tabpfn = True

print(f"\n--- 5. INFERENCE (Sampling {INFERENCE_SAMPLE_SIZE}) ---")

num_samples = min(INFERENCE_SAMPLE_SIZE, len(X_test))
sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
X_test_sample = np.ascontiguousarray(X_test[sample_indices])
y_test_sample = np.ascontiguousarray(y_test[sample_indices])


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pm1 = np.mean(np.abs(y_pred - y_true) <= 1)
    return acc, pm1


print("Predicting XGBoost...")
xgb_preds = xgb.predict(X_test_sample)
x_acc, x_pm1 = evaluate(y_test_sample, xgb_preds)

print("\n" + "=" * 60)
print(f"{'METRIC':<25} | {'XGBOOST':<10} | {'TABPFN':<10}")
print("=" * 60)

# 2. TabPFN gets a STRATIFIED SUBSET (Capped at 10,000)
if len(X_train_raw) > TABPFN_CONTEXT_SIZE:
    print(f"   📉 Downsampling TabPFN context: {len(X_train_raw)} -> {TABPFN_CONTEXT_SIZE} samples...")
    # Stratified downsample ensures we keep the class balance
    _, X_tabpfn, _, y_tabpfn = train_test_split(
        X_train_raw, y_train_raw,
        test_size=TABPFN_CONTEXT_SIZE,
        stratify=y_train_raw,
        random_state=42
    )
    X_tabpfn = np.ascontiguousarray(X_tabpfn, dtype=np.float32)
    y_tabpfn = np.ascontiguousarray(y_tabpfn)
else:
    print("   ✅ Dataset small enough for full TabPFN context.")
    X_tabpfn = X_test_sample
    y_tabpfn = y_test_sample

print(f"   XGBoost Training Size: {X_context.shape}")
print(f"   TabPFN Context Size:   {X_tabpfn.shape}")

print("Fitting TabPFN (MPS Mode)...")
pfn = TabPFNClassifier(device='mps', n_estimators=4, n_jobs=1)
pfn.fit(X_tabpfn, y_tabpfn)

print("Predicting TabPFN...")
pfn_preds = []
for i in tqdm(range(0, len(X_test_sample), INFERENCE_BATCH_SIZE)):
    batch = X_test_sample[i: i + INFERENCE_BATCH_SIZE]
    batch_preds = pfn.predict(batch)
    pfn_preds.extend(batch_preds)
pfn_preds = np.array(pfn_preds)[:len(y_test_sample)]

t_acc, t_pm1 = evaluate(y_test_sample, pfn_preds)

print(f"{'Exact Accuracy':<25} | {x_acc:.1%}     | {t_acc:.1%}")
print(f"{'+/- 1 Bin Accuracy':<25} | {x_pm1:.1%}     | {t_pm1:.1%}")

# Save comparison plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test_sample, xgb_preds), annot=True, fmt='d', cmap='Blues')
plt.title(f"XGBoost (Dense)\nAcc: {x_acc:.1%}")
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test_sample, pfn_preds), annot=True, fmt='d', cmap='Greens')
plt.title(f"TabPFN (Dense)\nAcc: {t_acc:.1%}")
plt.tight_layout()
plt.savefig('final_dense_results.png')
print("✅ Saved comparison to final_dense_results.png")

# ==========================================
# 8. SHAPLEY VALUES FOR TABPFN
# ==========================================
print("\n--- 6. SHAP EXPLANATION (TabPFN) ---")

# 1. Background Summary: Summarize training data to 50 weighted means (kmeans)
# Note: KernelExplainer works with model agnostic predict functions
# For TabPFN, we use predict_proba to get class probabilities

print("\n--- 6. SHAP EXPLANATION (TabPFN) ---")

# Create a small summary of the training data (Background)
background_summary = shap.kmeans(X_context, 50)

explainer = shap.KernelExplainer(pfn.predict_proba, background_summary)

# 2. Calculate SHAP values for a subset of the Test set
shap_sample_size = 50  # Keep small for speed
print(f"   Calculating SHAP values for {shap_sample_size} samples...")
X_shap_test = X_test_sample[:shap_sample_size]

# Run SHAP
shap_values = explainer.shap_values(X_shap_test, nsamples=50)

print("   Saving SHAP values to disk...")
np.save("shap_values.npy", shap_values)
np.save("X_shap.npy", X_shap_test)  # Save the data snippet too!

print("✅ Saved shap_values.npy and X_shap.npy")

# 3. Plotting for EACH Class
print("\n   Generating SHAP Summary Plots for ALL Classes...")

# Loop through every class index in the list
for class_idx, class_vals in enumerate(shap_values):
    print(f"     -> Plotting Class {class_idx}...")

    plt.figure()
    # Create the summary plot for this specific class
    shap.summary_plot(class_vals, X_shap_test, feature_names=feature_names, show=False)

    plt.title(f"TabPFN Feature Importance (Class {class_idx})")
    plt.tight_layout()

    # Save with a unique filename
    filename = f'tabpfn_shap_summary_class_{class_idx}.png'
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory
    print(f"        ✅ Saved to '{filename}'")
