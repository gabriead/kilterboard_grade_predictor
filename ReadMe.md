
---

# KilterGrade: Predicting Climbing Difficulty with Dense Statistical Features & Tabular Transformers

### 🎯 Project Overview

This project benchmarks **TabPFN (a Prior-Data Fitted Network)** against **XGBoost** for predicting the difficulty grades of **Kilterboard climbing routes**.

Climbing data is geometric, sparse, and highly subjective. Standard "bag of words" or sparse vector approaches fail because 95% of the board is empty space. This project demonstrates that by engineering **dense statistical features** that capture the "physics" of a climb (density, spread, crux magnitude), a **Transformer-based In-Context Learner** can outperform traditional Gradient Boosting.

### 📊 Key Results: The "Transformer Advantage"

In a strictly controlled experiment on a class-balanced dataset of "Golden Archetype" climbs, **TabPFN outperformed XGBoost** significantly.

|**Metric**|**XGBoost (Baseline)**|**TabPFN (Transformer)**|**Improvement**|
|---|---|---|---|
|**Exact Accuracy**|32.3%|**39.6%**|**+7.3%**|
|**±1 Bin Accuracy**|64.6%|**70.8%**|**+6.2%**|

**Interpretation:**

While XGBoost is excellent at finding simple thresholds (e.g., _"If Angle > 50°, grade is hard"_), it struggles with complex, global interactions in small datasets. TabPFN, leveraging its pre-trained priors, effectively treats the problem as a multi-class likelihood estimation, using the "context" of similar climbs to refine its prediction in a way decision trees cannot.

_Figure 1: Confusion matrices comparing XGBoost (left) vs. TabPFN (right). Note TabPFN's stronger diagonal concentration, indicating higher confidence in the correct grade bin._

---

### 🧠 Data & Feature Engineering Strategy

The core technical challenge was transforming a sparse "Bag of Holds" into a dense feature vector that retains spatial context.

#### 1. The "Golden Archetype" Dataset

To eliminate subjective labeling noise, we filtered the 300k+ climb database down to high-confidence samples:

- **Consensus:** Routes where `Display Grade == Average User Grade`.
    
- **Quality:** Only routes with a perfect 3-star rating.
    
- **Certainty:** No "soft" or "sandbagged" grades; distinct integers only.
    

#### 2. From Sparse to Dense: The "Physics" Vector

We moved from a 100-dimensional sparse vector (which resulted in 2.8% accuracy) to a **16-dimensional Dense Statistical Vector**:

|**Feature Group**|**Description**|**Why it matters**|
|---|---|---|
|**Global Geometry**|Wall Angle, Aspect Ratio, Board Area|Defines the physical "canvas" of the route.|
|**Distribution Stats**|Mean/Std Dev of X & Y coordinates|Captures the "center of mass" and "spread" (e.g., a wide traverse vs. a straight ladder).|
|**Role Counts**|Count of Hands vs. Feet|Distinguishes between "tracking" routes (many feet) and "power" routes (few feet).|
|**Physics Proxies**|**Crux Ratio**, **Max Move**, Density|Explicitly calculates the hardest single move and the sustained intensity.|

#### 3. Rigorous Class Balancing

To prevent the model from biased guessing (e.g., always predicting V4), we undersampled the majority classes to create a perfectly stratified dataset (N=60 per grade bin).

---

### 🔍 Interpretability & Insights

We used Feature Importance and t-SNE clustering to validate that the model is learning _climbing logic_, not just memorizing noise.

#### 1. Feature Importance Analysis

The model's top features align perfectly with human climbing intuition:

1. **Start Hold Role (`H2_Role`):** The hardest climbs often have specific, terrible starting holds. The model identified "How you start" as the #1 predictor.
    
2. **Wall Angle:** Basic physics—steeper is harder.
    
3. **Max Move (Crux):** The model heavily weighs the single largest distance between hands, validating the "Crux" concept.
    
4. **Std Dev X/Y:** The "spread" of holds indicates if a climb is awkward or straightforward.
    

#### 2. The "Grade Continuum" (t-SNE)

The t-SNE projection of our features shows a **smooth gradient** (rainbow) rather than distinct clusters.

- **Interpretation:** Climbing difficulty is continuous, not discrete. A hard V5 and an easy V6 are geometrically similar. The fact that the colors fade into each other proves our features capture the _spectrum_ of difficulty linearly.
    

---

### 🚀 Usage

Bash

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (ETL -> Training -> Eval -> Feature Importance)
# Note: TabPFN requires a CPU or MPS (Mac) environment.
python kilter_dense_tabpfn.py
```

### 🛠 Tech Stack

- **TabPFN:** Transformer-based In-Context Learning for tabular data.
    
- **XGBoost:** Gradient Boosting for baseline comparison.
    
- **Scikit-Learn:** Data stratification and t-SNE dimensionality reduction.
    
- **PyTorch / MPS:** Hardware acceleration for Transformer inference on Apple Silicon.
    
- **Pandas/NumPy:** Vectorized feature engineering.