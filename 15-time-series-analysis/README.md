# Module 15: Time Series Analysis

**Branch Module** | Est. time: 0.5–1 month (full-time) · 1–2 months (part-time)

*Best taken after Module 09 (Neural Networks), optionally in parallel with Modules 11/12*

---

## Learning Objectives

By the end of this module you will:
- Decompose time series and test for stationarity
- Fit ARIMA, SARIMA, and exponential smoothing models
- Build ML features from lag and rolling statistics
- Train LSTM and GRU networks for sequence forecasting
- Evaluate all models correctly with walk-forward validation

---

## Prerequisites

- Module 08: Unsupervised Learning; Module 09: Neural Networks Basics

---

## 1. Decomposition & Stationarity

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

df = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")
ts = df["sales"]

# ─── Decompose: trend + seasonality + residual ────────────────────────────────
result = seasonal_decompose(ts, model="additive", period=12)
result.plot(); plt.tight_layout(); plt.show()

# ─── Stationarity tests ───────────────────────────────────────────────────────
def check_stationarity(series):
    adf_result  = adfuller(series.dropna())
    kpss_result = kpss(series.dropna(), regression="c")
    print(f"ADF  p-value: {adf_result[1]:.4f}  (< 0.05 → stationary)")
    print(f"KPSS p-value: {kpss_result[1]:.4f}  (> 0.05 → stationary)")

check_stationarity(ts)

# ─── Make stationary: differencing ────────────────────────────────────────────
ts_diff  = ts.diff().dropna()           # 1st order difference
ts_diff2 = ts.diff().diff().dropna()    # 2nd order (rarely needed)
ts_log   = np.log1p(ts).diff().dropna() # Log + diff for multiplicative trends

# ─── ACF / PACF plots ─────────────────────────────────────────────────────────
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts_diff,  lags=40, ax=axes[0], title="ACF  (→ MA order q)")
plot_pacf(ts_diff, lags=40, ax=axes[1], title="PACF (→ AR order p)")
plt.tight_layout(); plt.show()
# Spikes at lag k in ACF → MA(k); spikes at lag k in PACF → AR(k)
```

---

## 2. Classical Models — ARIMA Family

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# ─── Manual ARIMA(p,d,q) ──────────────────────────────────────────────────────
model = ARIMA(train, order=(2, 1, 1))
fitted = model.fit()
print(fitted.summary())
forecast = fitted.forecast(steps=12)

# ─── Auto ARIMA (finds best p,d,q automatically) ─────────────────────────────
auto_model = auto_arima(train, seasonal=True, m=12,
                         stepwise=True, trace=True,
                         information_criterion="aic")
print(auto_model.summary())
forecast = auto_model.predict(n_periods=12)

# ─── SARIMA(p,d,q)(P,D,Q,m) — with seasonal component ───────────────────────
sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
fitted = sarima.fit(disp=False)
forecast = fitted.forecast(steps=12)

# ─── SARIMAX — with exogenous regressors ─────────────────────────────────────
sarima_x = SARIMAX(train, exog=train_exog,
                   order=(1,1,1), seasonal_order=(1,1,1,12))
fitted = sarima_x.fit(disp=False)
forecast = fitted.forecast(steps=12, exog=test_exog)

# ─── Holt-Winters Exponential Smoothing ───────────────────────────────────────
from statsmodels.tsa.holtwinters import ExponentialSmoothing
hw = ExponentialSmoothing(train, trend="add", seasonal="add",
                          seasonal_periods=12)
fitted = hw.fit()
forecast = fitted.forecast(12)
```

---

## 3. ML Features for Time Series

```python
import pandas as pd
import numpy as np

def create_features(df, target_col="sales", lags=[1,2,3,7,14,28],
                    rolling_windows=[7,14,28]):
    df = df.copy()

    # ─── Lag features ─────────────────────────────────────────────────────────
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # ─── Rolling statistics ────────────────────────────────────────────────────
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"]  = df[target_col].shift(1).rolling(w).std()
        df[f"rolling_max_{w}"]  = df[target_col].shift(1).rolling(w).max()
        df[f"rolling_min_{w}"]  = df[target_col].shift(1).rolling(w).min()

    # ─── Date/time features ────────────────────────────────────────────────────
    df["year"]       = df.index.year
    df["month"]      = df.index.month
    df["day"]        = df.index.day
    df["day_of_week"]= df.index.dayofweek
    df["quarter"]    = df.index.quarter
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    # ─── Cyclical encoding (sin/cos — no ordinal artifacts) ───────────────────
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df.dropna()

df_feat = create_features(df)

# Train LightGBM on features
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

feature_cols = [c for c in df_feat.columns if c != "sales"]
X, y = df_feat[feature_cols], df_feat["sales"]

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                           num_leaves=31, random_state=42)
model.fit(X.iloc[:-90], y.iloc[:-90])   # Hold out last 90 days for test
```

---

## 4. Walk-Forward (Rolling) Validation

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def walk_forward_validate(model, X, y, n_splits=5, horizon=30):
    """
    Proper time-series cross-validation: train on past → predict future.
    Never uses future data during training (no leakage).
    """
    n = len(X)
    fold_size = (n - horizon) // n_splits
    errors = []

    for i in range(n_splits):
        train_end   = fold_size * (i + 1)
        test_start  = train_end
        test_end    = min(train_end + horizon, n)

        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        mae  = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mape = np.mean(np.abs((y_te - preds) / (y_te + 1e-8))) * 100
        errors.append({"fold": i+1, "MAE": mae, "RMSE": rmse, "MAPE%": mape})
        print(f"Fold {i+1}: MAE={mae:.1f} | RMSE={rmse:.1f} | MAPE={mape:.1f}%")

    import pandas as pd
    result = pd.DataFrame(errors)
    print(f"\nMean  → MAE={result.MAE.mean():.1f} | RMSE={result.RMSE.mean():.1f} | MAPE={result['MAPE%'].mean():.1f}%")
    return result
```

---

## 5. Deep Sequence Models — LSTM / GRU

```python
import torch
import torch.nn as nn
import numpy as np

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # Use last time-step output

def create_sequences(data, seq_len=30, horizon=1):
    """Turn a 1D time series into supervised sequences."""
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + horizon])
    return np.array(X), np.array(y)

# Normalise
from sklearn.preprocessing import MinMaxScaler
scaler    = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()

SEQ_LEN = 30; HORIZON = 7
X, y = create_sequences(ts_scaled, SEQ_LEN, HORIZON)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)   # (N, seq, 1)
y = torch.tensor(y, dtype=torch.float32)

# Train
model     = LSTMForecaster(input_size=1, hidden_size=64, output_size=HORIZON)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X[:-50]), y[:-50])
    loss.backward(); optimizer.step()
    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X[-50:]), y[-50:])
        print(f"Epoch {epoch+1}: train={loss:.4f} | val={val_loss:.4f}")
```

---

## 6. Production Forecasting Frameworks

```python
# ─── Prophet (Meta) ────────────────────────────────────────────────────────────
from prophet import Prophet

prophet_df = df.reset_index().rename(columns={"date": "ds", "sales": "y"})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
            daily_seasonality=False, changepoint_prior_scale=0.05)
m.add_country_holidays(country_name="US")
m.fit(prophet_df[prophet_df.ds < "2025-01-01"])
future   = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
m.plot(forecast); m.plot_components(forecast)

# ─── NeuralForecast (GPU-accelerated deep learning) ──────────────────────────
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, PatchTST

nf = NeuralForecast(models=[NHITS(h=30, input_size=90, max_steps=500)],
                    freq="D")
nf.fit(df=train_nf_df)   # Expects columns: unique_id, ds, y
forecasts = nf.predict()
```

---

## Project: Sales Forecasting

**Dataset**: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) or M5 Forecasting

Steps:
1. EDA — trend, seasonality, holiday effects, store-level patterns
2. Stationarity tests + ACF/PACF analysis
3. Baseline: ARIMA / Holt-Winters
4. ML approach: LightGBM with lag + rolling features
5. Deep learning: LSTM forecaster
6. Compare all with walk-forward CV (5 folds, 30-day horizon)
7. Ensemble best two models: weighted average by inverse RMSE

**[← Module 14](../14-mlops-basics/README.md)** | **[→ Module 16](../16-projects-beginner/README.md)**
