# Module 04: Supervised Learning — Classification

**Phase 2 — ML Basics** | Est. time: 3–4 weeks (full-time) · 6–8 weeks (part-time)

---

## Prerequisites

- Module 03: Supervised Learning — Regression

---

## 1. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Binary classification
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]   # Probability of positive class

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Multiclass — one-vs-rest (default) or multinomial
lr_multi = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
```

**Decision boundary**: `P(y=1|x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)` where σ is the sigmoid.

---

## 2. K-Nearest Neighbours (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

# Tune k with cross-validation
from sklearn.model_selection import cross_val_score
import numpy as np

k_scores = {}
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="roc_auc")
    k_scores[k] = scores.mean()

best_k = max(k_scores, key=k_scores.get)
print(f"Best k: {best_k} → AUC: {k_scores[best_k]:.4f}")

knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn.fit(X_train, y_train)
```

⚠️ **Always scale features before KNN** — it's distance-based!

---

## 3. Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Gaussian NB: for continuous features (assumes Gaussian distribution)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Multinomial NB: for count data (e.g., word counts in text)
mnb = MultinomialNB(alpha=1.0)   # alpha = Laplace smoothing

# Bernoulli NB: for binary features (word present/absent)
bnb = BernoulliNB()
```

**When to use**: Fast, works with very small datasets, great for text classification baseline.

---

## 4. Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from sklearn import tree

dt = DecisionTreeClassifier(
    max_depth=5,           # Limit depth to prevent overfitting
    min_samples_split=20,  # Min samples to split a node
    min_samples_leaf=10,   # Min samples in leaf
    criterion="gini",      # "gini" or "entropy"
    random_state=42
)
dt.fit(X_train, y_train)

# Visualise tree structure
print(export_text(dt, feature_names=list(X_train.columns)))

# Feature importance (impurity-based)
import pandas as pd
imp = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(imp.head(10))
```

**Key hyperparameters**: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

---

## 5. Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel="linear", C=1.0, probability=True)

# RBF kernel (most common — maps to higher dimensions)
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
svm_rbf.fit(X_train, y_train)

y_prob = svm_rbf.predict_proba(X_test)[:, 1]
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

**SVM tips**: Scale features. RBF is default. C controls margin width (higher C = narrower margin, more overfitting). Slow on large datasets (>100K rows).

---

## 6. Classification Metrics

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Full report
print(classification_report(y_test, y_pred, target_names=["Retain", "Churn"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Precision: {tp/(tp+fp):.4f}  — of predicted positive, % correct")
print(f"Recall:    {tp/(tp+fn):.4f}  — of actual positive, % found")
print(f"F1:        {2*tp/(2*tp+fp+fn):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")

# ROC and PR curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax1)
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax2)
plt.tight_layout(); plt.show()
```

**Which metric to use?**

| Situation | Metric |
|---|---|
| Balanced classes | Accuracy, F1 |
| Imbalanced; FP costly | Precision |
| Imbalanced; FN costly | Recall |
| General ranking | ROC-AUC |
| Highly imbalanced | PR-AUC |

---

## 7. Bias Auditing with Fairlearn

```python
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, selection_rate

sensitive_features = X_test["gender"]

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)
print(mf.by_group)
```

---

## Project: Customer Churn Prediction

**Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Steps:
1. EDA — churn rate, feature distributions by churn group
2. Clean — `TotalCharges` to numeric, handle missing
3. Encode — OHE for categoricals
4. Train Logistic Regression, Decision Tree, Random Forest, SVM
5. Compare all models with 5-fold cross-validation
6. Identify most predictive features
7. Tune best model with GridSearchCV

**[← Module 03](../03-supervised-learning-regression/README.md)** | **[→ Module 05](../05-model-evaluation-optimization/README.md)**
