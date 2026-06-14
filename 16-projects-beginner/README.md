# Beginner Projects

**6 Projects** | Prerequisites: Phases 0–2 | Est. time per project: 2–5 days

Start these projects **during Phase 2** — do not wait until the end. Each project reinforces the module content and builds your portfolio.

---

## Project 1: House Price Prediction

**Skills**: Linear/Polynomial Regression, EDA, Feature Engineering, scikit-learn Pipelines  
**Dataset**: [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) — 79 features, 1,460 rows  
**Target metric**: RMSE on log-transformed sale price

**Steps**:
1. EDA — distribution of `SalePrice` (log-transform it), correlations, missing value heatmap
2. Impute: `LotFrontage` (median by neighbourhood), `GarageYrBlt` (fill with `YearBuilt`)
3. Engineer: `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`, `HouseAge`, `IsNew`
4. Encode: OrdinalEncoder for quality ratings (Po→1 to Ex→5), OHE for nominals
5. Train Ridge, Lasso, ElasticNet — tune α with 5-fold CV
6. Stretch: Add XGBoost and a stacking ensemble; aim for top 30% on Kaggle

**Key learning**: proper train/val/test split, sklearn Pipeline, log-transforming skewed targets

---

## Project 2: Iris Flower Classification

**Skills**: Multi-class classification, EDA, multiple algorithm comparison  
**Dataset**: `sklearn.datasets.load_iris()` — 150 samples, 4 features, 3 classes  
**Target metric**: 5-fold CV accuracy

**Steps**:
1. EDA — pairplot coloured by species, boxplots of each feature
2. Train 5 classifiers: Logistic Regression, KNN, Decision Tree, Random Forest, SVM
3. Cross-validate each with `StratifiedKFold(n_splits=5)`
4. Build a results DataFrame: model → [mean_acc, std_acc, train_time]
5. Visualise decision boundaries (2D slice using first two features)
6. Stretch: Add a voting ensemble of the top 3 models

**Key learning**: multi-classifier comparison methodology, cross-validation, result reporting

---

## Project 3: Titanic Survival Prediction

**Skills**: Binary classification, data cleaning, feature engineering, Kaggle submission  
**Dataset**: [Kaggle Titanic](https://www.kaggle.com/c/titanic) — 891 training rows  
**Target metric**: Kaggle accuracy (aim: > 78%)

**Steps**:
1. EDA — survival rate by Pclass, Sex, Age band, Embarked, FamilySize
2. Impute: Age → median by title (`Mr`, `Mrs`, `Miss`, `Master`), Embarked → mode
3. Engineer: `Title` (extract from Name), `FamilySize`, `IsAlone`, `FareBand`, `AgeBand`
4. Encode: `Sex` (0/1), `Embarked` (OHE), ordinal bands
5. Train Random Forest; tune with GridSearchCV
6. Submit to Kaggle; iterate to improve

**Key learning**: real-world data cleaning, Kaggle workflow, feature engineering from text

---

## Project 4: Spam Email Detection

**Skills**: Text classification, TF-IDF vectorisation, Naive Bayes, precision/recall tradeoff  
**Dataset**: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) — 5,572 messages  
**Target metric**: F1-score on spam class (positive class)

**Steps**:
1. EDA — class balance (~13% spam), message length by class, word clouds
2. Preprocess: lowercase, remove punctuation and stop words, lemmatise with spaCy
3. Vectorise: `TfidfVectorizer(ngram_range=(1,2), max_features=10000)`
4. Train Multinomial Naive Bayes, Logistic Regression, LinearSVC — compare F1
5. Plot confusion matrix and PR curve for best model
6. Stretch: Build a Streamlit app where users paste a message and see spam probability

**Key learning**: text preprocessing pipeline, TF-IDF, classification metrics beyond accuracy

---

## Project 5: Wine Quality Prediction

**Skills**: Regression + classification framing, feature importance, EDA  
**Dataset**: [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) — 6,497 wines, 11 features  
**Target metric**: RMSE (regression) or F1-macro (classification)

**Steps**:
1. EDA — distribution of quality scores (class imbalance!), correlation heatmap
2. Frame as both: regression (predict 0–10) and 3-class (low/medium/high)
3. Train Random Forest on both framings; compare results
4. Feature importance: which chemical properties drive quality?
5. Stretch: Compare Red vs. White wine models — do the same features matter?

**Key learning**: problem framing choice, feature importance interpretation

---

## Project 6: Customer Data Dashboard with Streamlit

**Skills**: Pandas, Plotly, Streamlit, interactive data apps, deployment  
**Dataset**: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) or any customer CSV  
**Target**: A deployed, interactive web app

**Steps**:
1. Load and clean customer data — handle missing `Income` (median), fix `Dt_Customer` to datetime
2. Feature engineer: `Age`, `TotalSpend`, `NumChildren`, `CustomerDuration`
3. Build Streamlit app with:
   - Sidebar filters (Age range slider, Income range, Education level)
   - KPI cards: Total Customers, Avg Spend, Avg Income, Avg Age
   - Bar chart: Spend by product category
   - Scatter plot: Income vs. TotalSpend coloured by Marital Status
   - Download button for filtered data as CSV
4. Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) — free
5. Stretch: Add a K-Means segment column and colour the scatter by segment

**Key learning**: data app development, interactive visualisation, free cloud deployment

---

## How to Structure Each Project

```
project-name/
├── README.md          ← Problem, approach, results, key learnings, live demo link
├── notebook.ipynb     ← Full analysis with markdown explanations
├── solution.py        ← Clean, documented script version
├── requirements.txt
└── data/
    └── README.md      ← How to download the dataset (don't commit data to Git!)
```

**README template for each project**:
- Problem: What are we predicting? Why does it matter?
- Dataset: Source, size, key features
- Approach: Which algorithms, which features, any special handling
- Results: Best metric achieved, comparison table of models tried
- Key learnings: What surprised you? What would you do differently?
- Live demo link (if deployed)

**[← Main README](../README.md)** | **[→ Intermediate Projects](../17-projects-intermediate/README.md)**
