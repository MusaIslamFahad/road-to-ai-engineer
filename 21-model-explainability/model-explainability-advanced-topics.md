# Model Explainability — Advanced Topics

---

## 1. Integrated Gradients (Deep Learning)

Integrated Gradients is an axiomatic attribution method that works for any differentiable model.

```python
import torch
import torch.nn as nn
import numpy as np

def integrated_gradients(model, input_tensor, baseline=None, n_steps=50, target_class=1):
    """
    Compute Integrated Gradients for a single input.
    
    baseline: the 'absence of information' reference (zeros by default)
    n_steps:  number of steps along the interpolation path
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Interpolate between baseline and input
    alphas  = torch.linspace(0, 1, n_steps + 1)
    interps = torch.stack([baseline + a * (input_tensor - baseline) for a in alphas])

    # Compute gradients at each interpolation step
    interps.requires_grad_(True)
    outputs = model(interps)[:, target_class]
    outputs.sum().backward()

    grads = interps.grad.detach()  # (n_steps+1, n_features)

    # Approximate integral using trapezoidal rule
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0)
    ig        = (input_tensor - baseline) * avg_grads

    return ig.detach().numpy()

# Usage
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

model = SimpleNet()
# ... train model ...

x_sample = torch.FloatTensor(X_test[0])
ig_attrs  = integrated_gradients(model, x_sample, target_class=1)

# Visualise
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
colors = ['red' if v < 0 else 'green' for v in ig_attrs]
plt.bar(feature_names, ig_attrs, color=colors)
plt.title('Integrated Gradients Attribution')
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('integrated_gradients.png', dpi=150)
```

---

## 2. SHAP Interaction Values

Interaction values measure how pairs of features interact in the model's predictions.

```python
import shap
import xgboost as xgb
import numpy as np

model = xgb.XGBClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)

# SHAP interaction values: shape (n_samples, n_features, n_features)
# [i, j, k]: contribution of feature j interacting with feature k for sample i
shap_interaction = explainer.shap_interaction_values(X_test[:200])
print(f"Interaction values shape: {shap_interaction.shape}")

# Mean absolute interaction strength matrix
mean_interactions = np.abs(shap_interaction).mean(axis=0)

import pandas as pd
import seaborn as sns

interaction_df = pd.DataFrame(mean_interactions,
                               index=feature_names,
                               columns=feature_names)

plt.figure(figsize=(12, 10))
sns.heatmap(interaction_df, annot=True, fmt='.3f', cmap='viridis')
plt.title('Mean |SHAP| Interaction Values')
plt.tight_layout()
plt.savefig('shap_interactions.png', dpi=150)

# Which pair has the strongest interaction?
np.fill_diagonal(mean_interactions, 0)  # exclude self-interactions
flat_idx = np.argmax(mean_interactions)
i, j     = np.unravel_index(flat_idx, mean_interactions.shape)
print(f"Strongest interaction: {feature_names[i]} × {feature_names[j]}"
      f" = {mean_interactions[i, j]:.4f}")
```

---

## 3. Global Surrogate Models

Train an interpretable model to approximate a black-box model globally.

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Train complex black-box model
black_box = GradientBoostingClassifier(n_estimators=500, max_depth=6, random_state=42)
black_box.fit(X_train, y_train)

# Get black-box predictions on a large sample
X_sample = np.vstack([X_train, X_test])
pseudo_labels = black_box.predict(X_sample)

# Train interpretable surrogate
surrogate = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
surrogate.fit(X_sample, pseudo_labels)

# Measure fidelity (how well does surrogate match black-box?)
fidelity = accuracy_score(pseudo_labels, surrogate.predict(X_sample))
print(f"Surrogate fidelity: {fidelity:.4f}")  # target > 0.90

# Print decision rules (fully interpretable)
rules = export_text(surrogate, feature_names=feature_names, max_depth=3)
print(rules)

# Surrogate makes feature importance interpretable
surrogate_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': surrogate.feature_importances_
}).sort_values('importance', ascending=False)
print(surrogate_importance.head(10))
```

---

## 4. Anchors (Rule-Based Local Explanations)

Anchors provide "if-then" rules that guarantee a prediction regardless of other features.

```python
from anchor import anchor_tabular

# Create anchor explainer
anchor_exp = anchor_tabular.AnchorTabularExplainer(
    class_names=['No Churn', 'Churn'],
    feature_names=feature_names,
    train_data=X_train,
    categorical_names={}   # dict of categorical feature indices → labels
)

# Explain a prediction
idx     = 42
exp     = anchor_exp.explain_instance(
    X_test[idx],
    model.predict,
    threshold=0.95   # rule must apply with 95% precision
)

print(f"Prediction: {['No Churn', 'Churn'][model.predict([X_test[idx]])[0]]}")
print(f"Anchor rule: IF {' AND '.join(exp.names())}")
print(f"Precision:   {exp.precision():.2f}")
print(f"Coverage:    {exp.coverage():.2f} ({exp.coverage()*100:.0f}% of data)")
```

---

## 5. Fairness-Aware Explainability

Check whether the model uses sensitive features differently across groups.

```python
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assume X_test_df has a 'gender' column
X_test_df = pd.DataFrame(X_test, columns=feature_names)
X_test_df['gender'] = gender_test  # 0=Female, 1=Male

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # positive class

# SHAP values per group
male_mask   = X_test_df['gender'] == 1
female_mask = X_test_df['gender'] == 0

male_shap   = np.abs(shap_values[male_mask]).mean(axis=0)
female_shap = np.abs(shap_values[female_mask]).mean(axis=0)

diff_df = pd.DataFrame({
    'feature':         feature_names,
    'male_importance': male_shap,
    'female_importance': female_shap,
    'ratio':           male_shap / (female_shap + 1e-8)
}).sort_values('ratio', ascending=False)

print("Features most differently weighted across gender:")
print(diff_df.head(10).to_string(index=False))

# Flag features with extreme disparity (ratio > 2x or < 0.5x)
flagged = diff_df[(diff_df['ratio'] > 2.0) | (diff_df['ratio'] < 0.5)]
if not flagged.empty:
    print(f"\n⚠️ {len(flagged)} features show >2× disparity across gender groups:")
    print(flagged[['feature','ratio']].to_string(index=False))
```

---

## 6. Concept-Based Explanations (TCAV)

TCAV tests whether a model uses human-defined concepts in its predictions.

```python
# Simplified TCAV implementation
from sklearn.linear_model import LogisticRegression
import numpy as np

def compute_cav(model_activations_positive, model_activations_negative):
    """
    Compute a Concept Activation Vector (CAV) using a linear probe.
    positive: activations of concept examples (e.g., images with stripes)
    negative: random other examples
    """
    X_cav = np.vstack([model_activations_positive, model_activations_negative])
    y_cav = np.array([1]*len(model_activations_positive) +
                     [0]*len(model_activations_negative))
    probe = LogisticRegression().fit(X_cav, y_cav)
    cav   = probe.coef_[0]   # direction in activation space
    return cav / np.linalg.norm(cav)

def tcav_score(model, layer_activations, cav, target_class, n_samples=100):
    """
    TCAV score: fraction of samples where increasing the concept
    increases the target class probability.
    """
    # Directional derivatives: dot product of gradient and CAV
    directional_derivs = []
    for act in layer_activations[:n_samples]:
        # Simplified: gradient of output w.r.t. activation
        grad = np.random.randn(*cav.shape)  # replace with actual gradient
        directional_derivs.append(np.dot(grad, cav))

    tcav = np.mean(np.array(directional_derivs) > 0)
    return tcav

# tcav > 0.5: concept is positively associated with the target class
```

---

## 7. Explaining Recommendations

```python
import shap
import numpy as np
import pandas as pd

# Matrix factorisation: user embeddings (n_users, k) and item embeddings (n_items, k)
# To explain why user u was recommended item i:

def explain_recommendation(user_id, item_id, user_emb, item_emb, feature_names_u,
                            feature_names_i, top_k=5):
    """
    Explain recommendation via dot product decomposition.
    Score = sum(user_emb[user_id] * item_emb[item_id])
    """
    u  = user_emb[user_id]
    it = item_emb[item_id]
    contributions = u * it  # element-wise product

    df = pd.DataFrame({
        'dimension': range(len(contributions)),
        'contribution': contributions
    }).sort_values('contribution', ascending=False)

    print(f"Recommendation score: {u.dot(it):.4f}")
    print(f"\nTop {top_k} dimensions driving recommendation:")
    print(df.head(top_k).to_string(index=False))
    print(f"\nBottom {top_k} dimensions opposing recommendation:")
    print(df.tail(top_k).to_string(index=False))
```

---

## 8. Model Cards Template (Regulatory)

```python
def generate_model_card(
    model_name, version, developer, use_case, out_of_scope,
    training_data_desc, evaluation_data_desc,
    performance_metrics, fairness_metrics, limitations, contact
):
    """Generate a standardised model card."""
    card = f"""
# Model Card: {model_name} v{version}

## Model Details
- **Developer**: {developer}
- **Version**: {version}
- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
- **Type**: {use_case}

## Intended Use
- **Primary use**: {use_case}
- **Out-of-scope**: {out_of_scope}

## Training Data
{training_data_desc}

## Evaluation Data
{evaluation_data_desc}

## Performance Metrics
{performance_metrics}

## Fairness Analysis
{fairness_metrics}

## Limitations
{limitations}

## Contact
{contact}
"""
    print(card)
    with open(f'model_card_{model_name.replace(" ","_")}.md', 'w') as f:
        f.write(card)

generate_model_card(
    model_name     = "Churn Predictor",
    version        = "2.1",
    developer      = "ML Platform Team",
    use_case       = "Predict probability of customer churn for retention campaigns",
    out_of_scope   = "Not for employment, credit, or housing decisions",
    training_data_desc = "85,000 customers from 2022-2024 CRM data",
    evaluation_data_desc = "15,000 held-out customers (same period)",
    performance_metrics = """| Metric | Overall | Male | Female |
|---|---|---|---|
| AUC-ROC | 0.924 | 0.921 | 0.928 |
| F1-Macro | 0.712 | 0.709 | 0.716 |""",
    fairness_metrics = "Demographic parity difference: 0.03 (< 0.1 threshold)",
    limitations     = "Lower performance for customers with < 3 months tenure",
    contact         = "ml-team@company.com"
)
```
