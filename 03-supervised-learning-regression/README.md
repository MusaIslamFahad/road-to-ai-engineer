# Module 03: Supervised Learning — Regression

**Phase 2 — ML Basics** | Est. time: 3–4 weeks (full-time) · 6–8 weeks (part-time)

---

## Learning Objectives

- Implement Simple and Multiple Linear Regression from scratch and with sklearn
- Apply Polynomial Regression for non-linear relationships
- Understand and apply L1 / L2 / ElasticNet regularization
- Run statistical regression analysis with statsmodels (p-values, confidence intervals)
- Implement all three flavours of gradient descent

---

## Prerequisites

- Module 02: Introduction to ML

---

## 1. Simple Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# y = β₀ + β₁x  (one feature)
X = np.array([[1400],[1600],[1700],[1875],[1100],[1550],[2350],[2450],[1425],[1700]])
y = np.array([245000,312000,279000,308000,199000,219000,405000,324000,319000,255000])

model = LinearRegression()
model.fit(X, y)

print(f"Intercept (β₀): {model.intercept_:,.0f}")
print(f"Coefficient (β₁): {model.coef_[0]:,.2f}  ← price per sq ft")
print(f"R²: {r2_score(y, model.predict(X)):.4f}")

# Predict new house
print(f"2000 sq ft house: ${model.predict([[2000]])[0]:,.0f}")
```

---

## 2. Multiple Linear Regression

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("housing.csv")   # Multiple features

X = df[["size", "bedrooms", "age", "garage", "school_rating"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)
print(f"RMSE: ${rmse:,.0f}  |  R²: {r2:.4f}")

# Coefficients (after scaling → comparable magnitudes)
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat:20s}: {coef:+.2f}")
```

---

## 3. Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Degree 2: adds x², x₁·x₂, etc.
poly_model = Pipeline([
    ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler",StandardScaler()),
    ("model", LinearRegression())
])

poly_model.fit(X_train, y_train)
preds = poly_model.predict(X_test)
print(f"Polynomial R²: {r2_score(y_test, preds):.4f}")

# ⚠ Warning: degree ≥ 3 overfits aggressively on small datasets
```

---

## 4. Regularization (Ridge, Lasso, ElasticNet)

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import numpy as np

# ─── Ridge (L2): shrinks all coefficients, keeps all features ────────────────
ridge = Ridge(alpha=1.0)   # alpha = λ (regularization strength)
ridge_cv = cross_val_score(ridge, X_train, y_train, cv=5, scoring="r2")
print(f"Ridge CV R²: {ridge_cv.mean():.4f} ± {ridge_cv.std():.4f}")

# ─── Lasso (L1): drives some coefficients exactly to zero (feature selection)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Non-zero features: {(lasso.coef_ != 0).sum()} / {len(lasso.coef_)}")

# ─── ElasticNet: combines L1 + L2 (great for correlated features) ────────────
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)

# ─── Tune alpha with cross-validation ────────────────────────────────────────
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_cv_model.fit(X_train, y_train)
print(f"Best alpha (Ridge): {ridge_cv_model.alpha_}")
```

**When to use which:**

| Regularizer | Formula | Use When |
|---|---|---|
| Ridge (L2) | λΣwᵢ² | All features are relevant; multicollinearity |
| Lasso (L1) | λΣ\|wᵢ\| | Automatic feature selection; sparse solution |
| ElasticNet | λ₁Σ\|wᵢ\| + λ₂Σwᵢ² | Correlated features; groups |

---

## 5. Gradient Descent

```python
import numpy as np

def gradient_descent(X, y, lr=0.01, n_iter=1000):
    """Simple batch gradient descent for linear regression."""
    m, n = X.shape
    # Add bias column
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.zeros(n + 1)
    history = []

    for i in range(n_iter):
        predictions = X_b @ theta
        error       = predictions - y
        gradients   = (2/m) * X_b.T @ error    # ∂MSE/∂θ
        theta      -= lr * gradients
        history.append(((error**2).mean()))     # Track MSE

    return theta, history

# Stochastic GD: update per sample (noisy but fast for large N)
def sgd(X, y, lr=0.01, n_epochs=50):
    m, n = X.shape
    X_b  = np.c_[np.ones((m,1)), X]
    theta = np.zeros(n+1)
    for epoch in range(n_epochs):
        idx = np.random.permutation(m)   # Shuffle each epoch
        for i in idx:
            xi, yi = X_b[i:i+1], y[i:i+1]
            gradient = 2 * xi.T @ (xi @ theta - yi)
            theta -= lr * gradient
    return theta

# Mini-batch GD: best of both (used in deep learning)
BATCH_SIZE = 32
```

---

## 6. Statistical Regression Analysis (statsmodels)

```python
import statsmodels.api as sm
import pandas as pd

X_sm = sm.add_constant(X_train_raw)   # Add intercept term
model_sm = sm.OLS(y_train, X_sm).fit()

print(model_sm.summary())
# Outputs: R², Adj. R², F-statistic, p-values per feature,
#          confidence intervals, Durbin-Watson (autocorrelation)

# Key values to check:
# p-value < 0.05 → feature is statistically significant
# VIF > 10       → multicollinearity concern
# Durbin-Watson ≈ 2 → no autocorrelation in residuals
```

---

## 7. Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def regression_report(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print(f"RMSE:  {rmse:,.2f}  (penalises large errors heavily)")
    print(f"MAE:   {mae:,.2f}   (robust to outliers)")
    print(f"R²:    {r2:.4f}   ({r2*100:.1f}% of variance explained)")
    print(f"MAPE:  {mape:.2f}%  (relative error — good for forecasting)")

regression_report(y_test, preds)
```

| Metric | Formula | When to Use |
|---|---|---|
| MSE | mean((y-ŷ)²) | Penalise outliers heavily |
| RMSE | √MSE | Same units as target |
| MAE | mean(\|y-ŷ\|) | Robust, interpretable |
| R² | 1 - SS_res/SS_tot | Percentage of variance explained |
| MAPE | mean(\|y-ŷ\|/y)×100 | Relative % error (forecasting) |

---

## Project: House Price Prediction

**Dataset**: [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Steps:
1. EDA — distribution of `SalePrice`, correlations with numeric features
2. Handle missing values (LotFrontage → median by neighbourhood, etc.)
3. Encode categoricals (OrdinalEncoder for quality ratings, OHE for others)
4. Feature engineering — TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
5. Log-transform skewed features and target
6. Train Ridge, Lasso, ElasticNet — compare with cross-validation
7. **Target**: top 30% on Kaggle leaderboard

**[← Module 02](../02-introduction-to-ml/README.md)** | **[→ Module 04](../04-supervised-learning-classification/README.md)**
