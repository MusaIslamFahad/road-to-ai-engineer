# Module 20: Handling Imbalanced Data

**Phase 7.5 — Essential Skills** | Est. time: 2–3 weeks (full-time) · 4–6 weeks (part-time)

*Best taken after Module 04 (Classification). Can be studied in parallel with any later module.*

---

## Learning Objectives

By the end of this module you will:
- Diagnose and quantify class imbalance in a dataset
- Apply oversampling (SMOTE, ADASYN), undersampling, and combined strategies
- Use algorithm-level techniques (class weights, scale_pos_weight)
- Tune decision thresholds for business-specific cost functions
- Evaluate imbalanced classifiers correctly (PR-AUC, F1, MCC — never raw accuracy)

---

## Prerequisites

- Module 04: Supervised Learning — Classification

---

## 1. Detecting and Quantifying Imbalance

```python
import pandas as pd
import matplotlib.pyplot as plt

# ─── Check class distribution ─────────────────────────────────────────────────
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True).round(4))

# Imbalance ratio
majority = df["target"].value_counts().max()
minority = df["target"].value_counts().min()
ratio    = majority / minority
print(f"Imbalance ratio: {ratio:.1f}:1")
# < 4:1   → mild     → class_weight="balanced" may suffice
# 4–30:1  → moderate → combine class weights + SMOTE
# > 30:1  → severe   → SMOTEENN / SMOTETomek + cost-sensitive learning
```

---

## 2. Resampling Techniques

```python
from imblearn.over_sampling  import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine        import SMOTEENN, SMOTETomek
import pandas as pd

# ─── SMOTE: create synthetic minority samples by interpolating between neighbours ──
smote = SMOTE(sampling_strategy="minority", k_neighbors=5, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"Before: {pd.Series(y_train).value_counts().to_dict()}")
print(f"After:  {pd.Series(y_res).value_counts().to_dict()}")

# ─── ADASYN: focuses on harder-to-learn minority samples ─────────────────────
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)

# ─── BorderlineSMOTE: oversample only near decision boundary ─────────────────
border_smote = BorderlineSMOTE(kind="borderline-1", random_state=42)

# ─── Random Undersampling: remove majority samples ───────────────────────────
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

# ─── Tomek Links: remove majority samples that are Tomek Links with minority ─
tomek = TomekLinks(sampling_strategy="majority")

# ─── SMOTEENN: SMOTE + Edited Nearest Neighbours cleaning ────────────────────
smoteenn = SMOTEENN(smote=SMOTE(k_neighbors=5, random_state=42), random_state=42)

# ─── SMOTETomek: SMOTE + Tomek Links cleaning ────────────────────────────────
smotetomek = SMOTETomek(random_state=42)
```

---

## 3. The Critical Rule: Resample Inside CV Folds

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ✅ CORRECT: SMOTE inside each fold — no leakage
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42)),
    ("clf",    RandomForestClassifier(n_estimators=200, n_jobs=-1))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(pipeline, X, y, cv=cv,
                         scoring=["roc_auc", "f1", "average_precision"])
print(f"PR-AUC: {results['test_average_precision'].mean():.4f} ± {results['test_average_precision'].std():.4f}")

# ❌ WRONG: SMOTE before split — test fold is contaminated
# X_res, y_res = SMOTE().fit_resample(X, y)   # Never do this!
# cross_val_score(model, X_res, y_res, cv=5)
```

---

## 4. Algorithm-Level Class Weighting

```python
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import xgboost as xgb
import lightgbm as lgb

# ─── Compute balanced weights ─────────────────────────────────────────────────
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
cw_dict = dict(zip(classes, weights))
print(f"Class weights: {cw_dict}")   # e.g., {0: 0.53, 1: 8.47}

# ─── sklearn models ───────────────────────────────────────────────────────────
lr  = LogisticRegression(class_weight="balanced", max_iter=1000)
rf  = RandomForestClassifier(class_weight="balanced", n_estimators=200)
svm = SVC(class_weight="balanced", probability=True)

# ─── XGBoost: scale_pos_weight = count(negative) / count(positive) ───────────
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=neg/pos, n_estimators=300,
                                eval_metric="aucpr", random_state=42)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)

# ─── LightGBM ────────────────────────────────────────────────────────────────
lgb_model = lgb.LGBMClassifier(is_unbalance=True, n_estimators=300,
                                 random_state=42)
# or: class_weight="balanced"
```

---

## 5. Decision Threshold Tuning

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_recall_curve,
                              precision_score, recall_score)

# Train model
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_val)[:, 1]

# ─── Method 1: maximise F1 on validation set ─────────────────────────────────
thresholds = np.arange(0.05, 0.95, 0.01)
f1_scores  = [f1_score(y_val, y_prob >= t, zero_division=0) for t in thresholds]
best_t_f1  = thresholds[np.argmax(f1_scores)]
print(f"Best threshold (max F1): {best_t_f1:.2f} → F1={max(f1_scores):.4f}")

# ─── Method 2: from Precision-Recall curve ───────────────────────────────────
precision, recall, pr_thresholds = precision_recall_curve(y_val, y_prob)
f1_from_pr = 2 * precision * recall / (precision + recall + 1e-8)
best_idx   = np.argmax(f1_from_pr)
best_t_pr  = pr_thresholds[best_idx]
print(f"Best threshold (PR curve): {best_t_pr:.2f}")

# ─── Method 3: business cost function ────────────────────────────────────────
# e.g., cost of FN (missed fraud) = $500, cost of FP (false alarm) = $5
cost_fn, cost_fp = 500, 5
costs = []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    fn = ((preds == 0) & (y_val == 1)).sum()
    fp = ((preds == 1) & (y_val == 0)).sum()
    costs.append(fn * cost_fn + fp * cost_fp)
best_t_cost = thresholds[np.argmin(costs)]
print(f"Best threshold (min cost): {best_t_cost:.2f} → Cost=${min(costs):,.0f}")

# ─── Apply optimal threshold ──────────────────────────────────────────────────
y_test_prob  = model.predict_proba(X_test)[:, 1]
y_test_pred  = (y_test_prob >= best_t_f1).astype(int)
```

---

## 6. Correct Evaluation Metrics

```python
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix,
    PrecisionRecallDisplay, RocCurveDisplay
)
import numpy as np

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ─── Full report ──────────────────────────────────────────────────────────────
print(classification_report(y_test, y_pred, target_names=["Majority", "Minority"]))

# ─── Key metrics for imbalanced data ─────────────────────────────────────────
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}  ← general ranking")
print(f"PR-AUC   : {average_precision_score(y_test, y_prob):.4f}  ← BEST for imbalanced")
print(f"F1 (min) : {f1_score(y_test, y_pred, pos_label=1):.4f}")
print(f"F1 macro : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"MCC      : {matthews_corrcoef(y_test, y_pred):.4f}  ← best single metric")

# ─── Confusion matrix details ─────────────────────────────────────────────────
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nSensitivity (Recall): {tp/(tp+fn):.4f}  — % minority found")
print(f"Specificity         : {tn/(tn+fp):.4f}  — % majority correct")
print(f"G-Mean              : {((tp/(tp+fn)) * (tn/(tn+fp)))**0.5:.4f}")
```

| Metric | For Imbalanced? | Why |
|---|---|---|
| **Accuracy** | ❌ Never | Misleading — 99% accuracy by predicting majority every time |
| **ROC-AUC** | ⚠️ Use carefully | Optimistic when imbalance is severe |
| **PR-AUC** | ✅ Best | Sensitive to minority class performance |
| **F1 (minority)** | ✅ Good | Balances precision and recall for minority |
| **MCC** | ✅ Best single | Uses all four confusion matrix cells |
| **G-Mean** | ✅ Good | Geometric mean of sensitivity & specificity |

---

## 7. Strategy Selection Guide

```
What's your imbalance ratio?
├── < 4:1  → class_weight="balanced" is enough
├── 4–30:1 → SMOTE (in Pipeline) + class_weight + threshold tuning
└── > 30:1 → SMOTEENN or SMOTETomek + scale_pos_weight + strict threshold

What's your model?
├── sklearn (LR, RF, SVM) → class_weight="balanced"
├── XGBoost              → scale_pos_weight = neg/pos
├── LightGBM             → is_unbalance=True
└── Neural Network       → pos_weight in BCEWithLogitsLoss

What metric matters most?
├── Recall (FN costly: cancer, fraud) → low threshold, high recall
├── Precision (FP costly: spam)       → high threshold, high precision
└── Both matter equally               → maximise F1 or G-Mean
```

---

## Related Resources

- [Imbalanced Data Cheatsheet](../resources/imbalanced_data_cheatsheet.md) — full code reference
- [Data Science Cheatsheet](../resources/data_science_cheatsheet.md)

**[← Module 19](../19-sql-database-fundamentals/README.md)** | **[→ Module 21](../21-model-explainability/README.md)**
