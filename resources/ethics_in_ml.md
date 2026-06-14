# Ethics in Machine Learning

A practical guide to bias, fairness, transparency, and responsible AI — for engineers building real systems.

---

## Why Ethics Matters for AI Engineers

You are not just a technical builder. Every ML system you deploy affects real people. Understanding ethics is:

- **Legally required** in many domains (EU AI Act, GDPR, ECOA, FHA)
- **Career-critical** — companies increasingly expect engineers to raise ethical concerns
- **Risk management** — biased or opaque models create legal, reputational, and financial risk
- **The right thing to do** — your models make decisions about people's loans, healthcare, hiring, and freedom

---

## Part 1: Types of Bias in ML

### Where Bias Enters the Pipeline

```
Data Collection → Feature Selection → Model Training → Evaluation → Deployment → Feedback Loop
      ↑                 ↑                  ↑               ↑            ↑              ↑
Historical         Proxy               Objective        Metric       Threshold    Amplification
  bias            features             function         choice        choice       over time
```

### Types of Bias

| Type | What It Means | Example |
|---|---|---|
| **Historical bias** | Training data reflects past discrimination | Hiring data trained on past decisions that excluded women |
| **Representation bias** | Underrepresented groups in training data | Facial recognition trained mostly on lighter-skinned faces |
| **Measurement bias** | Features measured differently across groups | Arrest records vs. actual crime (policed areas have more arrests) |
| **Aggregation bias** | One model for all groups when subgroups differ | Using one diabetes model for all ethnicities |
| **Evaluation bias** | Evaluating on non-representative benchmark | Testing a resume screener only on college graduates |
| **Deployment bias** | Model used for unintended purpose | Using a recidivism model for bail decisions |
| **Feedback loop bias** | Model predictions change future data | Predictive policing → more policing → more arrest data |
| **Proxy discrimination** | Protected attribute encoded in other features | Zip code as proxy for race in lending |

---

## Part 2: Fairness Definitions

There is no single universally "correct" fairness definition. Different definitions are often mathematically incompatible. You must choose based on your context and stakeholders.

### The Main Definitions

```python
# Assume: y = true label, ŷ = predicted label, ŷ_prob = predicted probability
# A = sensitive attribute (e.g., race, gender)

# ─── 1. Demographic Parity (Statistical Parity) ───────────────────────────────
# P(ŷ=1 | A=0) = P(ŷ=1 | A=1)
# Equal positive prediction rate across groups
# Use when: equal representation in outcomes matters (lending, hiring)
# Problem: ignores actual qualifications

# ─── 2. Equalised Odds ────────────────────────────────────────────────────────
# P(ŷ=1 | y=1, A=0) = P(ŷ=1 | y=1, A=1)  [equal TPR]
# P(ŷ=1 | y=0, A=0) = P(ŷ=1 | y=0, A=1)  [equal FPR]
# Equal recall and false alarm rate across groups
# Use when: both false positives and false negatives matter (medical, criminal justice)

# ─── 3. Equal Opportunity ─────────────────────────────────────────────────────
# P(ŷ=1 | y=1, A=0) = P(ŷ=1 | y=1, A=1)  [equal TPR only]
# Equal recall (true positive rate) across groups
# Use when: missing qualified candidates is the primary concern (hiring, loans)

# ─── 4. Predictive Parity (Calibration) ──────────────────────────────────────
# P(y=1 | ŷ=1, A=0) = P(y=1 | ŷ=1, A=1)  [equal precision]
# If the model says 70% probability, it should mean 70% for all groups
# Use when: you act on probability scores (risk scoring, insurance)

# ─── 5. Individual Fairness ───────────────────────────────────────────────────
# Similar individuals should receive similar predictions
# Hard to operationalise: requires a distance metric on individuals
```

### The Impossibility Result

**Chouldechova's theorem** (2017): When base rates differ across groups, it is mathematically impossible to simultaneously satisfy:
- Calibration (predictive parity)
- Equal false positive rates
- Equal false negative rates

You must choose which errors you prioritise. That is a **values decision**, not a technical one.

---

## Part 3: Measuring Fairness with Fairlearn

```python
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference, true_positive_rate
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Fit your model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Sensitive features
sensitive = X_test["gender"]   # or race, age_group, etc.

# ─── MetricFrame: per-group breakdown ─────────────────────────────────────────
mf = MetricFrame(
    metrics={
        "accuracy":       accuracy_score,
        "precision":      precision_score,
        "recall":         recall_score,
        "selection_rate": selection_rate,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive
)
print("Overall metrics:")
print(mf.overall)
print("\nPer-group metrics:")
print(mf.by_group)
print("\nDifference (max - min):")
print(mf.difference(method="between_groups"))

# ─── Fairness summary statistics ──────────────────────────────────────────────
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)
eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive)
print(f"\nDemographic Parity Difference: {dp_diff:.4f}  (0 = perfect fairness)")
print(f"Equalised Odds Difference:     {eo_diff:.4f}  (0 = perfect fairness)")
# Rule of thumb: |difference| < 0.1 is often considered acceptable
```

### Fairness-Aware Model Selection

```python
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

# ─── Post-processing: adjust decision thresholds per group ────────────────────
thresh_opt = ThresholdOptimizer(
    estimator=model,
    constraints=EqualizedOdds(),
    objective="accuracy_score",
    predict_method="predict_proba"
)
thresh_opt.fit(X_train, y_train, sensitive_features=X_train["gender"])
y_fair = thresh_opt.predict(X_test, sensitive_features=X_test["gender"])

# ─── In-processing: fairness constraint during training ──────────────────────
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=DemographicParity(),
    eps=0.01    # Constraint tolerance
)
mitigator.fit(X_train, y_train, sensitive_features=X_train["gender"])
y_fair2 = mitigator.predict(X_test)
```

---

## Part 4: Transparency & Explainability

### SHAP for Fairness Analysis

```python
import shap
import numpy as np
import pandas as pd

explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test)

# ─── Does the model rely on sensitive features? ───────────────────────────────
feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

sensitive_features = ["gender", "race", "age", "zip_code"]
for feat in sensitive_features:
    if feat in feature_importance["feature"].values:
        importance = feature_importance.loc[
            feature_importance["feature"] == feat, "mean_abs_shap"
        ].values[0]
        print(f"  {feat}: mean |SHAP| = {importance:.4f}")

# ─── Group-level SHAP: does model explain differently per group? ──────────────
male_mask   = X_test["gender"] == "Male"
female_mask = X_test["gender"] == "Female"

male_shap   = np.abs(shap_vals[male_mask]).mean(axis=0)
female_shap = np.abs(shap_vals[female_mask]).mean(axis=0)

diff_df = pd.DataFrame({
    "feature": X_test.columns,
    "male_importance":   male_shap,
    "female_importance": female_shap,
    "ratio": male_shap / (female_shap + 1e-10)
}).sort_values("ratio", ascending=False)
print(diff_df.head(10))  # Features most differently used across groups
```

---

## Part 5: Regulatory Framework

### EU AI Act (2024 — in force)

| Risk Level | Examples | Requirements |
|---|---|---|
| **Unacceptable** | Social scoring, mass surveillance, subliminal manipulation | **Banned outright** |
| **High Risk** | Hiring, credit scoring, education, healthcare, law enforcement | Transparency, human oversight, bias testing, documentation |
| **Limited Risk** | Chatbots, deepfakes | Must disclose AI involvement |
| **Minimal Risk** | Spam filters, recommendation systems | No specific requirements |

**For AI Engineers building high-risk systems**:
- Maintain technical documentation (training data, model card, testing results)
- Conduct conformity assessments before deployment
- Implement human oversight mechanisms
- Log decisions for audit
- Register in EU database

### GDPR (Article 22)

- Individuals have the **right to explanation** for fully automated decisions
- Right to **object** to profiling
- Special protections for sensitive categories (health, race, religion, political views)
- **Data minimisation**: don't collect more than necessary

### US Regulations

| Law | Domain | Key Requirement |
|---|---|---|
| **ECOA (Equal Credit Opportunity Act)** | Lending | No discrimination on protected classes; adverse action notices |
| **Fair Housing Act** | Housing / mortgages | No discriminatory advertising or decisions |
| **EEOC Guidelines** | Employment | Disparate impact — 80% rule for adverse impact |
| **HIPAA** | Healthcare | Privacy and security of protected health information |

---

## Part 6: Model Cards & Datasheets

### Model Card Template

```markdown
# Model Card: [Model Name]

## Model Details
- **Developer**: [Team]
- **Date**: [YYYY-MM-DD]
- **Version**: [v1.0]
- **Type**: Binary classifier (XGBoost)
- **License**: Internal

## Intended Use
- **Primary use**: Predict customer churn to trigger retention campaigns
- **Out-of-scope uses**: NOT for employment decisions, NOT for credit scoring

## Training Data
- Source: CRM database, 2022–2024
- Size: 85,000 customers
- Known limitations: Underrepresents customers < 25 years old (8% of data)

## Evaluation Data
- Held-out test set: 15,000 customers, same time period
- Evaluation date: 2024-11-01

## Performance
| Metric | Overall | Male | Female | Age < 30 | Age ≥ 30 |
|---|---|---|---|---|---|
| AUC-ROC | 0.924 | 0.921 | 0.928 | 0.891 | 0.932 |
| F1 | 0.712 | 0.709 | 0.716 | 0.668 | 0.724 |

## Ethical Considerations
- Model does not use gender, race, or age as direct features
- Contract type (proxy for socioeconomic status) is a significant predictor
- Lower performance on customers < 30 — monitor this group specifically

## Caveats
- Performance may degrade for customer segments acquired after 2024
- Do not use for customers with < 3 months tenure (data sparsity)

## Contact
ml-platform-team@company.com
```

---

## Part 7: Practical Checklist for AI Engineers

### Before Building
- [ ] Who is affected by this model? What are the consequences of errors?
- [ ] What data am I using? Does it reflect historical discrimination?
- [ ] What protected attributes might be proxied in my features?
- [ ] Is this the right problem to solve with ML? (Are there non-ML alternatives?)

### During Development
- [ ] Test performance separately for protected groups (gender, age, race, disability)
- [ ] Measure demographic parity difference and equalised odds difference
- [ ] Check if the model uses proxies for protected attributes (SHAP analysis)
- [ ] Review training data distribution — who is underrepresented?

### Before Deployment
- [ ] Write a Model Card documenting fairness evaluation results
- [ ] Is there a human review process for high-stakes decisions?
- [ ] Is there an appeal/redress mechanism for affected individuals?
- [ ] Legal review for high-risk applications (hiring, lending, healthcare)
- [ ] Is there ongoing monitoring for fairness drift?

### After Deployment
- [ ] Monitor fairness metrics per release (not just accuracy)
- [ ] Set up alerts for disparate impact threshold breaches
- [ ] Collect and review user complaints related to unfairness
- [ ] Regular retraining avoids amplifying feedback loop bias

---

## Resources

| Resource | Type |
|---|---|
| [Fairlearn documentation](https://fairlearn.org) | Python library |
| [AI Fairness 360 (IBM)](https://aif360.mybluemix.net) | Python library |
| [Partnership on AI](https://partnershiponai.org) | Guidelines |
| [EU AI Act full text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) | Regulation |
| [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) | Google Research paper |
| [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) | Microsoft Research paper |
| [The Alignment Problem](https://brianchristian.org/the-alignment-problem/) | Book (Brian Christian) |
| [Weapons of Math Destruction](https://weaponsofmathdestructionbook.com) | Book (Cathy O'Neil) |
| [Atlas of AI](https://katecrawford.net) | Book (Kate Crawford) |
