# Module 02: Introduction to Machine Learning

**Phase 2 — ML Basics** | Est. time: 3–4 weeks (full-time) · 6–8 weeks (part-time)

---

## Learning Objectives

By the end of this module you will:
- Understand the three main types of ML and when to use each
- Know the end-to-end ML workflow from raw data to deployed model
- Build and evaluate your first real ML model using scikit-learn
- Avoid the most common beginner mistakes (leakage, wrong metrics, overfitting)

---

## Prerequisites

- Module 01: Python for Data Science (NumPy, Pandas, Matplotlib)

---

## 1. What is Machine Learning?

Machine Learning is the field of enabling computers to **learn from data** rather than following explicit rules.

| Type | What It Learns | Examples |
|---|---|---|
| **Supervised** | Mapping X → y from labeled pairs | Spam detection, house prices, churn |
| **Unsupervised** | Structure from unlabeled data | Customer segments, anomalies, topics |
| **Reinforcement** | Actions via rewards/penalties | Game agents, robotics, recommendation |
| **Self-Supervised** | Labels from data itself | BERT, GPT, SimCLR |

---

## 2. The ML Workflow

```
1. Define Problem          → What are we predicting? What metric matters?
2. Collect Data            → Historical data, APIs, scraping, databases
3. Explore & Clean         → EDA, handle missing values, remove duplicates
4. Feature Engineering     → Encode, scale, create new features
5. Split Data              → Train / Validation / Test (never touch test early!)
6. Choose Algorithm        → Start simple, then iterate
7. Train Model             → fit() on training data only
8. Evaluate                → Measure on held-out validation set
9. Tune Hyperparameters    → Grid/random search, Optuna
10. Final Evaluation       → Run ONCE on test set and report
11. Deploy                 → Serve predictions (API, batch, embedded)
12. Monitor                → Track drift, retrain when performance degrades
```

---

## 3. Batch vs. Online Learning

| | **Batch Learning** | **Online Learning** |
|---|---|---|
| Training | On full dataset at once | One sample at a time (SGD) |
| Update frequency | Periodic retrain | Continuous |
| Memory | Must fit in RAM | Can handle infinite stream |
| Use case | Most ML models | Stock prices, recommendation |

---

## 4. Instance-Based vs. Model-Based

| | **Instance-Based** | **Model-Based** |
|---|---|---|
| How it works | Memorise examples, compare new inputs | Learn parameters from data |
| Examples | k-NN, kernel SVM | Linear Reg, Random Forest, Neural Nets |
| Prediction | Look up nearest stored examples | Compute f(X; θ) |

---

## 5. Descriptive Statistics for ML

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

# ─── Central tendency & spread ────────────────────────────────────────────────
print(df.describe(include='all'))

# ─── Missing values ───────────────────────────────────────────────────────────
print(df.isnull().sum())
print(df.isnull().mean() * 100)          # % missing per column

# ─── Class balance (classification) ──────────────────────────────────────────
print(df["target"].value_counts(normalize=True))

# ─── Correlation ──────────────────────────────────────────────────────────────
corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
```

---

## 6. Data Types in ML

| Type | Description | Example | Encoding |
|---|---|---|---|
| **Numerical continuous** | Any real value | Age, salary, temperature | Scale or use raw |
| **Numerical discrete** | Integers | Count, rank | Often use raw |
| **Nominal categorical** | No order | Color, city, country | One-Hot or Target |
| **Ordinal categorical** | Natural order | Rating (1–5), size (S/M/L) | Label or Ordinal |
| **Binary** | Two values | Yes/No, 0/1 | Use raw or map |
| **Text** | Free-form strings | Reviews, tweets | TF-IDF, Embeddings |
| **Image** | Pixel arrays | Photos, scans | Normalize, augment |
| **Time series** | Ordered by time | Stock prices, sensor data | Lag features, rolling |

---

## 7. Your First End-to-End ML Model

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# 2. Feature engineering (keep simple for now)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna("S", inplace=True)
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

features = ["Pclass", "Age", "Fare", "FamilySize", "IsAlone",
            "Sex_male", "Embarked_Q", "Embarked_S"]
X = df[features]
y = df["Survived"]

# 3. Split — stratify to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit + transform on train
X_test  = scaler.transform(X_test)        # transform only on test (prevent leakage)

# 5. Train
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

---

## 8. Common Beginner Mistakes

| Mistake | What Goes Wrong | Fix |
|---|---|---|
| **Data leakage** | Test data influences training → overoptimistic results | Always split first, fit scaler on train only |
| **Wrong metric** | Using accuracy on imbalanced data → misleading | Use F1, ROC-AUC, or PR-AUC for imbalanced |
| **No validation set** | Overfit to test set → can't trust results | Train / Val / Test split or K-Fold CV |
| **Skipping EDA** | Miss data quality issues, wrong assumptions | Always EDA before modelling |
| **Too complex too fast** | Hard to debug, slow iteration | Start with linear model as baseline |
| **Forgetting to scale** | Distance-based models break (KNN, SVM, LR) | StandardScaler inside a Pipeline |

---

## 9. No Free Lunch Theorem

> No single algorithm is best for every problem.

- Simple data with linear patterns → Logistic / Linear Regression
- Tabular structured data → XGBoost / LightGBM / Random Forest
- Images → CNNs
- Text / Sequences → Transformers
- Always establish a **simple baseline** first

---

## Project: First End-to-End ML Model

**Dataset**: Titanic (built-in on Kaggle)
**Goal**: Predict survival (binary classification)

Steps:
1. EDA — survival rate by sex, class, age group
2. Feature engineering — FamilySize, IsAlone, title extraction from Name
3. Handle missing values — Age (median by title), Embarked (mode)
4. Train Logistic Regression, Decision Tree, Random Forest
5. Compare with 5-fold cross-validation
6. Submit to Kaggle (target: >78% accuracy)

**Stretch**: Build a Streamlit app where a user enters passenger details and sees their survival probability.

---

## Key Libraries

```python
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import classification_report, roc_auc_score
```

---

## Related Resources

- [Data Science Cheatsheet](../resources/data_science_cheatsheet.md)
- [ML Glossary](../resources/ml_glossary.md)
- [Math Formulas](../resources/math_formulas.md)

**[← Module 01](../01-python-for-data-science/README.md)** | **[→ Module 03](../03-supervised-learning-regression/README.md)**
