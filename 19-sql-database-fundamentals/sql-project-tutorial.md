# SQL Project Tutorial: End-to-End ML Feature Engineering Pipeline

Build a complete SQL-driven feature engineering pipeline that takes raw transactional data and produces a model-ready dataset.

---

## Project Overview

**Goal**: Build a customer churn prediction dataset entirely in SQL, then train and evaluate an ML model in Python.

**Skills**: CTEs, window functions, aggregations, Python+SQL, feature engineering

**Dataset**: Synthetic e-commerce data (we'll generate it)

**Time**: 4–6 hours

---

## Step 1: Set Up the Database

```python
# setup_db.py
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

engine = create_engine("postgresql://user:password@localhost:5432/ecommerce")
# Or use SQLite for local testing:
# engine = create_engine("sqlite:///ecommerce.db")

def setup_database():
    """Create tables and seed synthetic data."""
    with engine.begin() as conn:
        # ── Create tables ───────────────────────────────────────────────────
        conn.execute(text("""
            DROP TABLE IF EXISTS events, orders, customers CASCADE;

            CREATE TABLE customers (
                id           SERIAL PRIMARY KEY,
                name         VARCHAR(100),
                email        VARCHAR(150) UNIQUE,
                country      VARCHAR(50),
                age          INTEGER,
                plan_type    VARCHAR(20),   -- 'free', 'basic', 'premium'
                signup_date  DATE,
                churned      BOOLEAN DEFAULT FALSE,
                churn_date   DATE
            );

            CREATE TABLE orders (
                id           SERIAL PRIMARY KEY,
                customer_id  INTEGER REFERENCES customers(id),
                amount       DECIMAL(10,2),
                product_cat  VARCHAR(50),
                order_date   DATE,
                is_refunded  BOOLEAN DEFAULT FALSE
            );

            CREATE TABLE events (
                id           SERIAL PRIMARY KEY,
                customer_id  INTEGER REFERENCES customers(id),
                event_type   VARCHAR(50),   -- 'login', 'search', 'view', 'support_ticket'
                event_date   DATE
            );
        """))

    # ── Generate synthetic data ─────────────────────────────────────────────
    np.random.seed(42)
    random.seed(42)
    N_CUSTOMERS = 5000

    countries   = ['USA','UK','Canada','Germany','Australia']
    plans       = ['free','basic','premium']
    categories  = ['Electronics','Books','Clothing','Sports','Home']
    event_types = ['login','search','view','support_ticket','download']

    customers = []
    for i in range(N_CUSTOMERS):
        signup   = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730))
        churned  = random.random() < 0.25
        churn_d  = signup + timedelta(days=random.randint(90, 600)) if churned else None
        customers.append({
            'name':        f'Customer {i}',
            'email':       f'user{i}@example.com',
            'country':     random.choice(countries),
            'age':         random.randint(18, 70),
            'plan_type':   random.choice(plans),
            'signup_date': signup.date(),
            'churned':     churned,
            'churn_date':  churn_d.date() if churn_d else None,
        })

    df_customers = pd.DataFrame(customers)
    df_customers.to_sql('customers', engine, if_exists='append', index=False)

    # Generate orders
    orders = []
    for cid in range(1, N_CUSTOMERS + 1):
        n_orders = random.randint(0, 25)
        for _ in range(n_orders):
            order_dt = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 900))
            orders.append({
                'customer_id': cid,
                'amount':      round(random.uniform(5, 500), 2),
                'product_cat': random.choice(categories),
                'order_date':  order_dt.date(),
                'is_refunded': random.random() < 0.05,
            })
    pd.DataFrame(orders).to_sql('orders', engine, if_exists='append', index=False)

    # Generate events
    events = []
    for cid in range(1, N_CUSTOMERS + 1):
        n_events = random.randint(5, 200)
        for _ in range(n_events):
            event_dt = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 900))
            events.append({
                'customer_id': cid,
                'event_type':  random.choice(event_types),
                'event_date':  event_dt.date(),
            })
    pd.DataFrame(events).to_sql('events', engine, if_exists='append', index=False)

    print("✅ Database set up with:")
    print(f"   {N_CUSTOMERS} customers")
    print(f"   {len(orders)} orders")
    print(f"   {len(events)} events")

setup_database()
```

---

## Step 2: Exploratory SQL Analysis

```sql
-- How many customers and churn rate?
SELECT
    COUNT(*)                                          AS total_customers,
    SUM(churned::INT)                                 AS churned,
    ROUND(AVG(churned::FLOAT) * 100, 1)              AS churn_rate_pct
FROM customers;

-- Churn rate by plan type
SELECT
    plan_type,
    COUNT(*)                                          AS customers,
    ROUND(AVG(churned::FLOAT) * 100, 1)              AS churn_rate_pct
FROM customers
GROUP BY plan_type
ORDER BY churn_rate_pct DESC;

-- Order distribution
SELECT
    COUNT(*)                                          AS total_orders,
    COUNT(DISTINCT customer_id)                       AS customers_with_orders,
    ROUND(AVG(amount), 2)                            AS avg_order_value,
    ROUND(STDDEV(amount), 2)                         AS std_order_value,
    MIN(amount)                                       AS min_order,
    MAX(amount)                                       AS max_order
FROM orders
WHERE NOT is_refunded;

-- Most common event types
SELECT event_type, COUNT(*) AS n
FROM events
GROUP BY event_type
ORDER BY n DESC;

-- Customers with no orders (at risk)
SELECT COUNT(*) AS customers_no_orders
FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

---

## Step 3: Feature Engineering in SQL

This is the core step — building a rich feature table entirely in SQL.

```sql
-- feature_engineering.sql
-- Run with: psql -f feature_engineering.sql  OR  pd.read_sql(open(...).read(), engine)

CREATE OR REPLACE VIEW ml_features AS
WITH
-- ── Observation point: predict churn as of 2024-01-01 ────────────────────────
obs AS (SELECT DATE '2024-01-01' AS obs_date),

-- ── Order features ────────────────────────────────────────────────────────────
order_features AS (
    SELECT
        o.customer_id,
        COUNT(*)                                          AS order_count,
        COUNT(*) FILTER (WHERE o.is_refunded)            AS refund_count,
        ROUND(AVG(o.amount)::NUMERIC, 2)                 AS avg_order_value,
        ROUND(STDDEV(o.amount)::NUMERIC, 2)              AS std_order_value,
        SUM(o.amount) FILTER (WHERE NOT o.is_refunded)   AS total_spend,
        MAX(o.order_date)                                AS last_order_date,
        MIN(o.order_date)                                AS first_order_date,
        MAX(o.order_date) - MIN(o.order_date)            AS purchase_span_days,
        -- Recency: days since last order
        (SELECT obs_date FROM obs) - MAX(o.order_date)   AS days_since_last_order,
        -- Orders in last 90 days
        COUNT(*) FILTER (
            WHERE o.order_date >= (SELECT obs_date FROM obs) - INTERVAL '90 days'
        )                                                AS orders_last_90d,
        -- Spend in last 90 days
        SUM(o.amount) FILTER (
            WHERE o.order_date >= (SELECT obs_date FROM obs) - INTERVAL '90 days'
            AND NOT o.is_refunded
        )                                                AS spend_last_90d
    FROM orders o, obs
    WHERE o.order_date < obs.obs_date    -- only use data before observation point
    GROUP BY o.customer_id
),

-- ── Event features ────────────────────────────────────────────────────────────
event_features AS (
    SELECT
        e.customer_id,
        COUNT(*)                                          AS total_events,
        COUNT(*) FILTER (WHERE e.event_type = 'login')   AS login_count,
        COUNT(*) FILTER (WHERE e.event_type = 'support_ticket') AS support_tickets,
        COUNT(*) FILTER (WHERE e.event_type = 'search')  AS search_count,
        COUNT(DISTINCT e.event_date)                     AS active_days,
        MAX(e.event_date)                                AS last_event_date,
        (SELECT obs_date FROM obs) - MAX(e.event_date)   AS days_since_last_event,
        -- Events in last 30 days
        COUNT(*) FILTER (
            WHERE e.event_date >= (SELECT obs_date FROM obs) - INTERVAL '30 days'
        )                                                AS events_last_30d
    FROM events e, obs
    WHERE e.event_date < obs.obs_date
    GROUP BY e.customer_id
),

-- ── Rolling trend: is engagement increasing or decreasing? ───────────────────
trend_features AS (
    SELECT
        o.customer_id,
        -- Compare last 90 days vs 90-180 days ago
        COALESCE(SUM(o.amount) FILTER (
            WHERE o.order_date BETWEEN (SELECT obs_date FROM obs) - INTERVAL '90 days'
            AND (SELECT obs_date FROM obs)), 0
        ) AS spend_q1,
        COALESCE(SUM(o.amount) FILTER (
            WHERE o.order_date BETWEEN (SELECT obs_date FROM obs) - INTERVAL '180 days'
            AND (SELECT obs_date FROM obs) - INTERVAL '90 days'), 0
        ) AS spend_q2
    FROM orders o, obs
    GROUP BY o.customer_id
)

-- ── Final feature table ───────────────────────────────────────────────────────
SELECT
    c.id                                                      AS customer_id,
    -- Customer attributes
    c.age,
    EXTRACT(DAY FROM (SELECT obs_date FROM obs) - c.signup_date)::INT AS account_age_days,
    CASE c.plan_type WHEN 'premium' THEN 2 WHEN 'basic' THEN 1 ELSE 0 END AS plan_tier,
    CASE c.country  WHEN 'USA' THEN 1 ELSE 0 END             AS is_usa,
    -- Order features (with COALESCE for customers with no orders)
    COALESCE(of.order_count, 0)                               AS order_count,
    COALESCE(of.refund_count, 0)                              AS refund_count,
    COALESCE(of.avg_order_value, 0)                          AS avg_order_value,
    COALESCE(of.std_order_value, 0)                          AS std_order_value,
    COALESCE(of.total_spend, 0)                              AS total_spend,
    COALESCE(of.days_since_last_order, 9999)                 AS days_since_last_order,
    COALESCE(of.orders_last_90d, 0)                          AS orders_last_90d,
    COALESCE(of.spend_last_90d, 0)                           AS spend_last_90d,
    COALESCE(of.purchase_span_days, 0)                       AS purchase_span_days,
    -- Event features
    COALESCE(ef.total_events, 0)                             AS total_events,
    COALESCE(ef.login_count, 0)                              AS login_count,
    COALESCE(ef.support_tickets, 0)                          AS support_tickets,
    COALESCE(ef.active_days, 0)                              AS active_days,
    COALESCE(ef.days_since_last_event, 9999)                 AS days_since_last_event,
    COALESCE(ef.events_last_30d, 0)                          AS events_last_30d,
    -- Trend: is spend going up or down?
    CASE
        WHEN COALESCE(tf.spend_q2, 0) = 0 THEN 0
        ELSE ROUND(((tf.spend_q1 - tf.spend_q2) / NULLIF(tf.spend_q2, 0))::NUMERIC, 3)
    END                                                       AS spend_trend,
    -- Derived ratios
    CASE WHEN COALESCE(of.order_count, 0) = 0 THEN 0
         ELSE ROUND(COALESCE(of.refund_count, 0)::NUMERIC / of.order_count, 3)
    END                                                       AS refund_rate,
    -- Target label
    c.churned::INT                                            AS label
FROM customers c, obs
LEFT JOIN order_features  of ON c.id = of.customer_id
LEFT JOIN event_features  ef ON c.id = ef.customer_id
LEFT JOIN trend_features  tf ON c.id = tf.customer_id
WHERE c.signup_date < obs.obs_date;    -- only customers who signed up before obs date
```

---

## Step 4: Load Features into Python and Train

```python
# train.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

engine = create_engine("postgresql://user:password@localhost:5432/ecommerce")

# ── Load feature table ────────────────────────────────────────────────────────
df = pd.read_sql("SELECT * FROM ml_features", engine)
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['label'].mean():.2%}")

# ── Prepare features and target ───────────────────────────────────────────────
feature_cols = [c for c in df.columns if c not in ['customer_id', 'label']]
X = df[feature_cols]
y = df['label']

print("\nFeature types:")
print(X.dtypes.value_counts())
print(f"\nMissing values: {X.isnull().sum().sum()}")

# ── Train/test split (stratified) ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── Model 1: Random Forest ────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"\nRandom Forest CV AUC: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

# ── Model 2: XGBoost ──────────────────────────────────────────────────────────
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=neg/pos, eval_metric='auc',
    random_state=42, n_jobs=-1
)
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"XGBoost CV AUC:       {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")

# ── Model 3: LightGBM ────────────────────────────────────────────────────────
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    is_unbalance=True, random_state=42, n_jobs=-1
)
lgb_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"LightGBM CV AUC:      {lgb_scores.mean():.4f} ± {lgb_scores.std():.4f}")

# ── Final evaluation on test set ──────────────────────────────────────────────
best_model = lgb_model  # pick the best from CV
best_model.fit(X_train, y_train)
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

print("\n── Test Set Results ──")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:   {average_precision_score(y_test, y_prob):.4f}")

# ── Feature importance ────────────────────────────────────────────────────────
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

importance.plot(kind='barh', x='feature', y='importance', ax=axes[0],
                title='Top 15 Feature Importances')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'])
axes[1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('results.png', dpi=150)
print("\n✅ Results saved to results.png")
```

---

## Step 5: Write Predictions Back to SQL

```python
# write_predictions.py
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine("postgresql://user:password@localhost:5432/ecommerce")

# Generate predictions for all customers
df_all = pd.read_sql("SELECT * FROM ml_features", engine)
X_all  = df_all[[c for c in df_all.columns if c not in ['customer_id', 'label']]]
probs  = best_model.predict_proba(X_all)[:, 1]

# Build predictions DataFrame
preds = pd.DataFrame({
    'customer_id':    df_all['customer_id'],
    'churn_prob':     probs.round(4),
    'risk_tier':      pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                             labels=['low', 'medium', 'high']),
    'predicted_at':   pd.Timestamp.now()
})

# Write to database
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS churn_predictions"))
preds.to_sql('churn_predictions', engine, if_exists='replace', index=False)
print(f"✅ Wrote {len(preds)} predictions to churn_predictions table")

# Verify
result = pd.read_sql("""
    SELECT risk_tier, COUNT(*) AS n,
           ROUND(AVG(churn_prob) * 100, 1) AS avg_prob_pct
    FROM churn_predictions
    GROUP BY risk_tier ORDER BY avg_prob_pct DESC
""", engine)
print(result.to_string(index=False))
```

---

## Step 6: Business Dashboard Query

```sql
-- Who to target for retention campaigns?
SELECT
    c.name, c.email, c.plan_type, c.country,
    cp.churn_prob, cp.risk_tier,
    f.days_since_last_order,
    f.total_spend,
    f.support_tickets
FROM churn_predictions cp
JOIN customers c ON cp.customer_id = c.id
JOIN ml_features f ON cp.customer_id = f.customer_id
WHERE cp.risk_tier = 'high'
  AND c.churned = FALSE             -- still active
  AND f.total_spend > 200           -- valuable customers
ORDER BY cp.churn_prob DESC
LIMIT 100;
```

---

## Project Wrap-Up

By the end of this project you have:

1. ✅ Created a real relational database with 3 linked tables
2. ✅ Performed EDA in SQL to understand the data
3. ✅ Built a rich feature table using CTEs, window functions, and aggregations
4. ✅ Loaded features into Python and trained 3 ML models
5. ✅ Compared models with cross-validation and evaluated on a held-out test set
6. ✅ Written predictions back to the database
7. ✅ Built a business-ready targeting query

**Stretch goals:**
- Add EXPLAIN ANALYZE to your feature query and optimise slow parts
- Add more features: preferred product category, hour-of-day activity patterns
- Tune XGBoost or LightGBM with Optuna to squeeze more AUC
- Build a Streamlit app that queries `churn_predictions` and shows a dashboard
