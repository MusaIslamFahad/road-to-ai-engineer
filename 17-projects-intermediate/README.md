# Intermediate Projects

**8 Projects** | Prerequisites: Phases 0–5 | Est. time per project: 4–7 days

Start these during **Phase 5 (Deep Learning Fundamentals)** and complete them through Phase 6.

---

## Project 1: Handwritten Digit Recognition (MNIST)

**Skills**: CNNs, PyTorch DataLoader, GPU training, confusion matrix analysis  
**Dataset**: `torchvision.datasets.MNIST` — 70,000 images, 28×28 pixels  
**Target metric**: Test accuracy > 99%

**Steps**:
1. Build a CNN: Conv→ReLU→MaxPool→Conv→ReLU→MaxPool→FC→FC
2. Use `DataLoader` with `batch_size=64`, `shuffle=True`
3. Train with Adam + CosineAnnealingLR for 20 epochs
4. Analyse errors: plot the confusion matrix, visualise the most-confused digit pairs
5. Stretch: Try transfer learning with a small ResNet; compare to your CNN

**Architecture hint**:
```python
nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 10)
)
```

---

## Project 2: Customer Churn Prediction

**Skills**: Imbalanced data, SMOTE, threshold tuning, SHAP, business metric optimisation  
**Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers  
**Target metric**: PR-AUC + F1 on churn class

**Steps**:
1. EDA — churn rate (~27%), distribution by contract type, tenure, charges
2. Clean: `TotalCharges` → numeric, impute 11 missing values
3. Encode: OHE for categoricals; scale numerics
4. Handle imbalance: compare (a) class_weight="balanced", (b) SMOTE inside Pipeline
5. Tune XGBoost with Optuna (50 trials); optimise PR-AUC
6. Find optimal decision threshold on validation set
7. SHAP beeswarm plot — explain which features drive churn most
8. Stretch: Build a risk-tier system: Low / Medium / High risk buckets

---

## Project 3: Movie Recommendation System

**Skills**: Collaborative filtering, matrix factorisation, content-based, hybrid  
**Dataset**: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1M ratings  
**Target metric**: RMSE on held-out ratings; NDCG@10

**Steps**:
1. EDA — rating distribution, movies per user, users per movie, sparsity
2. User-based collaborative filtering with cosine similarity
3. Matrix factorisation with SVD (`scipy.sparse.linalg.svds`)
4. Content-based: TF-IDF on genres + title → item embeddings → cosine sim
5. Hybrid: weighted average of CF and content-based scores
6. Evaluate with RMSE and compare all four approaches
7. Stretch: Implement a simple neural collaborative filtering (NCF) model in PyTorch

---

## Project 4: Credit Card Fraud Detection

**Skills**: Extreme class imbalance (0.17% fraud), anomaly detection, PR curve  
**Dataset**: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions  
**Target metric**: PR-AUC (accuracy is meaningless here)

**Steps**:
1. EDA — class distribution (492 fraud vs. 284,315 normal), feature distributions
2. Scale `Amount` and `Time` (V1–V28 already PCA-reduced)
3. Train Logistic Regression, Random Forest, XGBoost — each with `class_weight="balanced"`
4. Also try Isolation Forest as unsupervised anomaly detector
5. Plot PR curves for all supervised models
6. Tune decision threshold to maximise recall at precision > 85%
7. Stretch: Compare SMOTE vs. class weights on XGBoost — which wins on PR-AUC?

---

## Project 5: Customer Segmentation

**Skills**: K-Means, DBSCAN, PCA, UMAP, cluster profiling, business interpretation  
**Dataset**: [Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) or e-commerce RFM data  
**Target metric**: Silhouette score + business interpretability

**Steps**:
1. EDA — age, income, spending score distributions; pairplot
2. Elbow method + silhouette score to choose optimal k
3. K-Means clustering; also try HDBSCAN (detects arbitrary shapes)
4. Reduce to 2D with UMAP for visualisation; colour by cluster
5. Profile each cluster: mean age, income, spend per cluster
6. Give each segment a name: "High-value loyals", "Budget shoppers", etc.
7. Stretch: Build an RFM (Recency/Frequency/Monetary) table and segment on that

---

## Project 6: Time Series Sales Forecasting

**Skills**: ARIMA, LSTM, walk-forward CV, feature engineering for time series  
**Dataset**: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) or any monthly sales CSV  
**Target metric**: MAPE and RMSE with walk-forward validation (5 folds, 30-day horizon)

**Steps**:
1. EDA — trend, seasonality, holiday effects, store-to-store variation
2. Stationarity test (ADF); ACF/PACF for order selection
3. Baseline: Seasonal Naïve (last year same period)
4. ARIMA / SARIMA with auto_arima
5. LightGBM with lag features and rolling statistics
6. LSTM forecaster in PyTorch
7. Compare all on walk-forward CV; build a results table
8. Stretch: Prophet with holiday regressors

---

## Project 7: Feature Engineering Mastery

**Skills**: Advanced encoding, interaction features, Pipeline, target leakage prevention  
**Dataset**: Any Kaggle tabular competition (Tabular Playground Series works well)  
**Target metric**: Top 30% on Kaggle leaderboard

**Steps**:
1. Build a raw baseline: OHE + StandardScaler + LogisticRegression → note score
2. Add target encoding for high-cardinality categoricals (inside CV fold!)
3. Add polynomial interaction features for top numerical features
4. Add datetime decomposition if date columns exist
5. Feature selection: remove features with near-zero SHAP importance
6. Compare: baseline vs. engineered features on same model
7. Track all experiments in MLflow

---

## Project 8: Ensemble Methods Comparison

**Skills**: Random Forest, XGBoost, LightGBM, CatBoost, stacking, voting  
**Dataset**: [Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) or any binary classification  
**Target metric**: ROC-AUC with 5-fold CV

**Steps**:
1. Tune each base model with Optuna (30 trials each): RF, XGBoost, LightGBM, CatBoost
2. Soft-voting ensemble of all four
3. Stacking: use 5-fold OOF predictions as meta-features → LogisticRegression meta-learner
4. Build a comprehensive results table: model → [val_auc ± std, fit_time, inference_time]
5. Analyse feature importance across all models — do they agree?
6. Stretch: Add a neural network as a 5th base model

---

## Project Submission Checklist

For each project before adding to your portfolio:
- [ ] README covers: problem, dataset, approach, results, learnings
- [ ] No data committed to GitHub (add to `.gitignore`)
- [ ] Notebook runs top-to-bottom without errors (restart kernel → run all)
- [ ] Test set evaluation reported (not validation!)
- [ ] At least one visualisation (plot, chart, dashboard)
- [ ] Requirements.txt or environment.yaml included

**[← Beginner Projects](../16-projects-beginner/README.md)** | **[→ Advanced Projects](../18-projects-advanced/README.md)**
