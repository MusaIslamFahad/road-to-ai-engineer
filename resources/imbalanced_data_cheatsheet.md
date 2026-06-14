# Imbalanced Data Cheatsheet

Quick reference for handling class imbalance in ML classification problems.

---

## Detecting Imbalance

```python
import pandas as pd

# Check class distribution
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True))

# Imbalance ratio
majority = df["target"].value_counts().max()
minority = df["target"].value_counts().min()
ratio = majority / minority
print(f"Imbalance ratio: {ratio:.0f}:1")

# Visual
df["target"].value_counts().plot(kind="bar", title="Class Distribution")
```

**Rule of Thumb**:
- < 4:1 → mild imbalance, most algorithms handle it natively
- 4:1 – 30:1 → moderate imbalance, apply class weights or resampling
- > 30:1 → severe imbalance, combine multiple techniques

---

## Resampling Techniques

```python
from imblearn.over_sampling  import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine        import SMOTEENN, SMOTETomek
from imblearn.pipeline       import Pipeline as ImbPipeline

# ─── Oversampling ─────────────────────────────────────────────────────────────
# SMOTE: create synthetic minority samples by interpolating between neighbors
smote = SMOTE(sampling_strategy="minority", k_neighbors=5, random_state=42)

# ADASYN: like SMOTE but focuses on harder-to-learn minority samples
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)

# Borderline-SMOTE: only oversample near the decision boundary
border_smote = BorderlineSMOTE(kind="borderline-1", random_state=42)

X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"Before: {y_train.value_counts().to_dict()}")
print(f"After:  {pd.Series(y_res).value_counts().to_dict()}")

# ─── Undersampling ────────────────────────────────────────────────────────────
# Random: remove random majority samples
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

# Tomek Links: remove majority samples that form Tomek Links with minority
tomek = TomekLinks(sampling_strategy="majority")

# ENN: remove majority samples misclassified by their k nearest neighbors
enn = EditedNearestNeighbours(sampling_strategy="majority")

# ─── Combined ─────────────────────────────────────────────────────────────────
# SMOTEENN: SMOTE then ENN cleaning
smoteenn = SMOTEENN(smote=SMOTE(k_neighbors=5), enn=EditedNearestNeighbours())

# SMOTETomek: SMOTE then Tomek Links cleaning
smotetomek = SMOTETomek(random_state=42)

X_res, y_res = smoteenn.fit_resample(X_train, y_train)
```

---

## Algorithm-Level Approaches

### Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute balanced weights
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(class_weight_dict)   # {0: 0.53, 1: 8.47} — minority gets higher weight

# Apply to sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

rf  = RandomForestClassifier(class_weight="balanced", n_estimators=200)
lr  = LogisticRegression(class_weight="balanced", max_iter=1000)
svm = SVC(class_weight="balanced", probability=True)

# XGBoost: scale_pos_weight = count_negative / count_positive
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=neg/pos)

# LightGBM: is_unbalance or class_weight
lgb_model = lgb.LGBMClassifier(is_unbalance=True)
# or
lgb_model = lgb.LGBMClassifier(class_weight="balanced")
```

### Threshold Tuning

```python
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

# Train model with default or balanced weights
model.fit(X_train, y_train)
probs = model.predict_proba(X_val)[:, 1]

# Method 1: Grid search over thresholds
best_threshold, best_f1 = 0.5, 0
for thresh in np.arange(0.1, 0.9, 0.01):
    preds = (probs >= thresh).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1, best_threshold = f1, thresh

print(f"Best threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")

# Method 2: Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_val, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx]

# Apply optimal threshold
final_preds = (model.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)
```

---

## Correct Cross-Validation with Imbalanced Data

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

# ✅ CORRECT: Resampling inside each fold (no leakage)
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42)),
    ("clf",    RandomForestClassifier(n_estimators=100))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(
    pipeline, X, y, cv=cv,
    scoring=["roc_auc", "f1", "average_precision"],
    return_train_score=False
)

for metric, values in results.items():
    if "test_" in metric:
        print(f"{metric}: {values.mean():.4f} ± {values.std():.4f}")

# ❌ WRONG: Resample BEFORE splitting (data leakage)
# X_res, y_res = SMOTE().fit_resample(X, y)  # Don't do this!
# cross_val_score(model, X_res, y_res, cv=5)
```

---

## Evaluation Metrics for Imbalanced Data

```python
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]

# ─── Full Report ──────────────────────────────────────────────────────────────
print(classification_report(y_test, y_pred, target_names=["Majority", "Minority"]))

# ─── Key Metrics ──────────────────────────────────────────────────────────────
print(f"ROC-AUC:           {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC (Avg Prec): {average_precision_score(y_test, y_prob):.4f}")
print(f"F1 (minority):     {f1_score(y_test, y_pred, pos_label=1):.4f}")
print(f"F1 (macro):        {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"MCC:               {matthews_corrcoef(y_test, y_pred):.4f}")

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Sensitivity (Recall): {tp/(tp+fn):.4f}")
print(f"Specificity:          {tn/(tn+fp):.4f}")
print(f"G-Mean:               {((tp/(tp+fn)) * (tn/(tn+fp)))**0.5:.4f}")

# ─── PR Curve ─────────────────────────────────────────────────────────────────
PrecisionRecallDisplay.from_predictions(y_test, y_prob)
plt.title("Precision-Recall Curve")
plt.show()
```

---

## When to Use What

| Scenario | Recommended Approach |
|---|---|
| Mild imbalance (< 4:1) | `class_weight="balanced"` or nothing |
| Moderate imbalance (4:1–30:1) | SMOTE inside Pipeline + `class_weight` |
| Severe imbalance (> 30:1) | SMOTEENN or SMOTETomek + cost-sensitive learning |
| Very few minority samples (< 100) | ADASYN or Borderline-SMOTE; also consider data collection |
| Business cost of FN >> FP (fraud, medical) | Threshold tuning for high recall + class weights |
| Business cost of FP >> FN | Threshold tuning for high precision |
| Tree-based models | `class_weight="balanced"` + permutation importance |
| XGBoost | `scale_pos_weight=neg/pos` + PR-AUC evaluation |
| LightGBM | `is_unbalance=True` or `class_weight="balanced"` |
| Neural networks | `pos_weight` in `BCEWithLogitsLoss` (PyTorch) |

### PyTorch Weighted Loss for Imbalanced

```python
import torch
import torch.nn as nn

# Compute positive class weight
n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# This tells the model: false negatives are n_neg/n_pos times as costly as false positives
```

---

## Quick Decision Guide

```
Data: imbalanced?
  → Yes → How severe?
      → Mild (< 4:1) → class_weight="balanced" → DONE
      → Moderate → SMOTE in Pipeline + balanced weights → threshold tune
      → Severe (> 30:1) → SMOTEENN + cost-sensitive + tune threshold for recall

Evaluate with:
  → PR-AUC (primary for imbalanced)
  → F1 macro / F1 minority class
  → MCC (best single metric for severely imbalanced)
  → G-Mean (geometric mean of sensitivity and specificity)
  → NOT accuracy (misleading for imbalanced!)
```
