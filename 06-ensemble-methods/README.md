# Module 06: Ensemble Methods

**Phase 3 — Advanced Supervised ML** | Est. time: 2–3 weeks (full-time) · 4–6 weeks (part-time)

---

## Prerequisites

- Module 05: Model Evaluation & Optimization

---

## 1. Bagging — Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Classification
rf = RandomForestClassifier(
    n_estimators=500,        # More trees = more stable (diminishing returns after ~300)
    max_depth=None,          # Let trees grow deep — forest handles variance
    min_samples_leaf=5,      # Regularise: min samples per leaf
    max_features="sqrt",     # sqrt(n_features) for classification; n_features/3 for regression
    n_jobs=-1,               # Use all CPU cores
    random_state=42,
    oob_score=True           # Out-of-bag score (free validation estimate)
)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")   # ~= cross-val score without extra computation

# Feature importance
import pandas as pd
feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feat_imp.head(10))
```

---

## 2. Boosting — AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost

```python
# ─── Gradient Boosting (sklearn) — good baseline ─────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8
)

# ─── XGBoost — fast, regularised, handles missing values ─────────────────────
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,          # L1 and L2 regularisation
    eval_metric="auc", use_label_encoder=False,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,      # Stop when val AUC stops improving
              verbose=False)

# ─── LightGBM — fastest, great for large datasets ────────────────────────────
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1
)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

# ─── CatBoost — best native categorical support ───────────────────────────────
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=6,
    cat_features=["contract_type", "payment_method"],   # Pass categoricals directly!
    eval_metric="AUC", random_seed=42, verbose=100
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
```

---

## 3. Stacking & Voting Ensembles

```python
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ─── Voting: majority vote (hard) or average probability (soft) ──────────────
voting = VotingClassifier(
    estimators=[("rf", RandomForestClassifier(n_estimators=200)),
                ("xgb", xgb.XGBClassifier(n_estimators=200)),
                ("lgb", lgb.LGBMClassifier(n_estimators=200))],
    voting="soft",       # Average probabilities — usually better
    n_jobs=-1
)

# ─── Stacking: meta-learner trains on base model out-of-fold predictions ─────
stacking = StackingClassifier(
    estimators=[("rf",  RandomForestClassifier(n_estimators=200)),
                ("xgb", xgb.XGBClassifier(n_estimators=200)),
                ("lgb", lgb.LGBMClassifier(n_estimators=200))],
    final_estimator=LogisticRegression(C=0.1),  # Meta-learner
    cv=5,              # OOF predictions with 5-fold CV
    passthrough=False  # Don't pass original features to meta-learner
)
stacking.fit(X_train, y_train)
```

---

## 4. Comparison Table

| Method | Speed | Memory | Best For | Tuning Effort |
|---|---|---|---|---|
| Random Forest | Medium | High | Stable baseline, feature importance | Low |
| XGBoost | Fast | Medium | Tabular competitions, flexibility | High |
| LightGBM | Fastest | Low | Large datasets (>100K rows) | High |
| CatBoost | Medium | Medium | High-cardinality categoricals | Low |
| Stacking | Slow | High | Squeezing final performance | Very High |

---

## Project: Ensemble Methods Comparison

**Dataset**: Any Kaggle tabular competition (e.g., Tabular Playground Series)

Steps:
1. Train RF, XGBoost, LightGBM, CatBoost independently
2. Tune each with Optuna (50 trials each)
3. Build a soft-voting ensemble
4. Build a stacking ensemble
5. Compare all models with 5-fold CV and make a results table

**[← Module 05](../05-model-evaluation-optimization/README.md)** | **[→ Module 07](../07-feature-engineering/README.md)**
