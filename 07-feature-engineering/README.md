# Module 07: Feature Engineering

**Phase 3 — Advanced Supervised ML** | Est. time: 2–3 weeks (full-time) · 4–6 weeks (part-time)

---

## Prerequisites

- Module 06: Ensemble Methods

---

## 1. Feature Selection

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier

# ─── Filter: remove low-variance features ─────────────────────────────────────
vt = VarianceThreshold(threshold=0.01)   # Remove features with var < 1%
X_filtered = vt.fit_transform(X_train)

# ─── Filter: univariate statistical test ─────────────────────────────────────
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]

# ─── Wrapper: Recursive Feature Elimination ───────────────────────────────────
rfe = RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=15)
rfe.fit(X_train, y_train)
print(X_train.columns[rfe.support_].tolist())

# ─── Embedded: SHAP-based feature selection ───────────────────────────────────
import shap
model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_train)
mean_shap = np.abs(shap_vals).mean(axis=0) if shap_vals.ndim == 2 else np.abs(shap_vals[0]).mean(axis=0)
feature_importance = pd.Series(mean_shap, index=X_train.columns).sort_values(ascending=False)
top_features = feature_importance[feature_importance > 0.01].index.tolist()
```

---

## 2. Categorical Encoding

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder, BinaryEncoder, WOEEncoder

df = pd.DataFrame({"city": ["NY","LA","NY","Chicago","LA","NY"],
                   "size": ["S","M","L","M","XL","S"],
                   "churn": [1,0,1,0,1,0]})

# ─── One-Hot Encoding (nominal, low cardinality < 10 values) ─────────────────
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_ohe = ohe.fit_transform(df[["city"]])

# ─── Ordinal Encoding (ordinal categories) ────────────────────────────────────
ord_enc = OrdinalEncoder(categories=[["S","M","L","XL"]])
df["size_enc"] = ord_enc.fit_transform(df[["size"]])

# ─── Target Encoding (high cardinality, mean-encode with smoothing) ───────────
# ⚠ MUST be done inside CV folds to prevent leakage!
te = TargetEncoder(cols=["city"], smoothing=10)

# ─── Binary Encoding (high cardinality, saves memory vs OHE) ─────────────────
be = BinaryEncoder(cols=["city"])

# ─── Frequency Encoding ───────────────────────────────────────────────────────
freq_map = df["city"].value_counts(normalize=True).to_dict()
df["city_freq"] = df["city"].map(freq_map)

# ─── WOE Encoding (Weight of Evidence — for credit scoring) ──────────────────
woe = WOEEncoder(cols=["city"])
X_woe = woe.fit_transform(df[["city"]], df["churn"])
```

---

## 3. Scaling & Transformation

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)
from scipy import stats

# ─── StandardScaler: mean=0, std=1 — most common ────────────────────────────
ss = StandardScaler()

# ─── MinMaxScaler: [0, 1] range — good for neural networks ──────────────────
mm = MinMaxScaler()

# ─── RobustScaler: uses median and IQR — robust to outliers ─────────────────
rs = RobustScaler()

# ─── Log transform: compress right-skewed distributions ─────────────────────
df["log_income"] = np.log1p(df["income"])   # log1p handles zeros

# ─── Box-Cox: parametric transform for positive data ────────────────────────
pt_bc = PowerTransformer(method="box-cox")   # Requires positive values

# ─── Yeo-Johnson: like Box-Cox but handles zero and negative values ──────────
pt_yj = PowerTransformer(method="yeo-johnson")

# Check skewness before/after
print(f"Before: {df['income'].skew():.2f}")
df["income_transformed"] = pt_yj.fit_transform(df[["income"]])
print(f"After:  {df['income_transformed'].skew():.2f}")
```

---

## 4. PCA for Dimensionality Reduction

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fit PCA
pca = PCA(n_components=0.95, random_state=42)   # Keep 95% of variance
X_pca = pca.fit_transform(X_train_scaled)

print(f"Original: {X_train_scaled.shape[1]} features")
print(f"After PCA: {X_pca.shape[1]} components")
print(f"Variance explained: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

# Scree plot
plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.axhline(0.95, color="r", linestyle="--", label="95% threshold")
plt.xlabel("Number of Components"); plt.ylabel("Cumulative Variance Explained")
plt.title("PCA Scree Plot"); plt.legend(); plt.grid(True); plt.show()
```

---

## 5. sklearn Pipeline & ColumnTransformer

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

num_cols = ["age", "income", "tenure_months"]
cat_cols = ["contract_type", "payment_method"]

# ─── Preprocessing sub-pipelines ─────────────────────────────────────────────
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ─── ColumnTransformer: apply different pipelines to different columns ────────
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
], remainder="drop")

# ─── Full pipeline ────────────────────────────────────────────────────────────
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(n_estimators=200, n_jobs=-1))
])

# Train and evaluate — preprocessing is correctly applied inside each CV fold
full_pipeline.fit(X_train, y_train)
print(f"Test AUC: {roc_auc_score(y_test, full_pipeline.predict_proba(X_test)[:,1]):.4f}")

# ─── Custom transformer ───────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X["year"]  = pd.to_datetime(X["date"]).dt.year
        X["month"] = pd.to_datetime(X["date"]).dt.month
        X["dow"]   = pd.to_datetime(X["date"]).dt.dayofweek
        return X.drop(columns=["date"])
```

---

## Related Resources

- [Data Science Cheatsheet](../resources/data_science_cheatsheet.md)

**[← Module 06](../06-ensemble-methods/README.md)** | **[→ Module 08](../08-unsupervised-learning/README.md)**
