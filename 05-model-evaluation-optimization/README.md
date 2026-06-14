# Module 05: Model Evaluation & Optimization

**Phase 2 — ML Basics** | Est. time: 3–4 weeks (full-time) · 6–8 weeks (part-time)

---

## Prerequisites

- Module 04: Supervised Learning — Classification

---

## 1. Train / Validation / Test Split

```python
from sklearn.model_selection import train_test_split

# Two-stage split: train 70%, val 15%, test 15%
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15/0.85,
    random_state=42, stratify=y_train_full
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ⚠️ THE GOLDEN RULE: Touch the test set ONCE — at final evaluation only.
```

---

## 2. Cross-Validation

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Always use a Pipeline so preprocessing is re-fitted per fold
pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())])

# ─── Stratified K-Fold (default for classification) ───────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(pipe, X, y, cv=cv,
                         scoring=["roc_auc", "f1_macro", "accuracy"],
                         return_train_score=True)

for metric in ["test_roc_auc", "test_f1_macro"]:
    print(f"{metric}: {results[metric].mean():.4f} ± {results[metric].std():.4f}")

# ─── Time Series Split (walk-forward — no future leakage) ────────────────────
tscv = TimeSeriesSplit(n_splits=5, gap=0)

# ─── Group K-Fold (ensure same group never in both train and test) ───────────
gkf = GroupKFold(n_splits=5)
# cross_validate(pipe, X, y, cv=gkf, groups=patient_ids)
```

---

## 3. Data Leakage: Detection & Prevention

```python
# ─── Leakage Type 1: Scaling before split ─────────────────────────────────────
# ❌ WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # Uses test distribution!
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ CORRECT: fit scaler inside cross-validation via Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

# ─── Leakage Type 2: Target statistics before split ──────────────────────────
# ❌ WRONG
df["mean_target_by_city"] = df.groupby("city")["target"].transform("mean")

# ✅ CORRECT: use leave-one-out target encoding inside fold
from category_encoders import TargetEncoder
enc = TargetEncoder(cols=["city"])  # Handles encoding properly inside Pipeline

# ─── Leakage Type 3: Future data in time series ───────────────────────────────
# ❌ WRONG: rolling mean includes future rows
df["rolling_avg"] = df["value"].rolling(7).mean()

# ✅ CORRECT: shift to use only past data
df["rolling_avg_past"] = df["value"].shift(1).rolling(7).mean()
```

**Warning signs of leakage**: Suspiciously high accuracy (>99%?), a single feature with near-perfect correlation to target, model scores drop sharply in production.

---

## 4. Hyperparameter Tuning

```python
# ─── Grid Search: exhaustive (small search space) ────────────────────────────
from sklearn.model_selection import GridSearchCV

param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_leaf": [1, 5, 10],
}
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc",
                           n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# ─── Random Search: faster for large spaces ──────────────────────────────────
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "clf__n_estimators": randint(100, 1000),
    "clf__max_depth": [None, 5, 10, 15, 20, 30],
    "clf__min_samples_leaf": randint(1, 20),
    "clf__max_features": uniform(0.2, 0.8),
}
rand_search = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5,
                                 scoring="roc_auc", n_jobs=-1, random_state=42)
rand_search.fit(X_train, y_train)

# ─── Optuna: Bayesian optimisation (best for expensive models) ───────────────
import optuna
from sklearn.ensemble import GradientBoostingClassifier

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
    }
    model = GradientBoostingClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=-1)
print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 5. Bias-Variance Tradeoff & Learning Curves

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv, scoring="roc_auc", n_jobs=-1
    )
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train AUC", color="blue")
    plt.fill_between(train_sizes,
                     train_scores.mean(1) - train_scores.std(1),
                     train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="blue")
    plt.plot(train_sizes, val_scores.mean(axis=1),  label="Val AUC",   color="orange")
    plt.fill_between(train_sizes,
                     val_scores.mean(1) - val_scores.std(1),
                     val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="orange")
    plt.xlabel("Training set size"); plt.ylabel("AUC")
    plt.title("Learning Curves"); plt.legend(); plt.grid(True)
    plt.show()

# Diagnosing from curves:
# Large gap (train >> val): high variance → add data or regularise
# Both curves low and converged: high bias → more complex model or features
```

---

## 6. Model Calibration

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

# Train raw model (probabilities may be poorly calibrated)
from sklearn.ensemble import RandomForestClassifier
raw_model = RandomForestClassifier(n_estimators=200)
raw_model.fit(X_train, y_train)

# Calibrate with Platt Scaling (sigmoid) or Isotonic Regression
calibrated = CalibratedClassifierCV(raw_model, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)

# Visualise calibration
fig, ax = plt.subplots(figsize=(8, 6))
CalibrationDisplay.from_estimator(raw_model, X_test, y_test, ax=ax, label="Raw RF")
CalibrationDisplay.from_estimator(calibrated, X_test, y_test, ax=ax, label="Calibrated RF")
plt.title("Calibration Curve (Reliability Diagram)"); plt.show()
```

**Why calibration matters**: A model with AUC 0.85 but poorly calibrated probabilities will make bad decisions in systems that use the raw probability (risk scoring, fraud thresholds).

---

## Related Resources

- [Data Science Cheatsheet](../resources/data_science_cheatsheet.md)
- [Math Formulas](../resources/math_formulas.md) — evaluation metric formulas

**[← Module 04](../04-supervised-learning-classification/README.md)** | **[→ Module 06](../06-ensemble-methods/README.md)**
