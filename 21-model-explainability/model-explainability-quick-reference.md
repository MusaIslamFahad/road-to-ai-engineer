# Model Explainability — Quick Reference

---

## Method Selection Guide

```
What do you need?
├── Global importance (overall feature ranking)
│   ├── Tree model  → shap.TreeExplainer + shap.summary_plot(plot_type='bar')
│   ├── Any model   → permutation_importance (sklearn)
│   └── Linear      → |coefficients| (after scaling)
│
├── Local explanation (why THIS prediction?)
│   ├── Tree model  → shap.plots.waterfall(explainer(X)[i])
│   ├── Any model   → LimeTabularExplainer.explain_instance()
│   └── Neural net  → Integrated Gradients or SHAP DeepExplainer
│
├── Feature effect (how does changing X affect output?)
│   ├── 1D effect   → PartialDependenceDisplay.from_estimator(kind='average')
│   ├── Per-sample  → kind='both' (ICE + PDP)
│   └── Interaction → 2D PDP: features=[('f1','f2')]
│
├── Image explanation
│   ├── CNN         → Grad-CAM (gradient-based, fast)
│   └── Any model   → LIME for Images (slower, perturbation-based)
│
└── Text explanation
    ├── Fine-tuned transformer → shap.Explainer (Text masker)
    └── Any text classifier   → LimeTextExplainer
```

---

## SHAP One-Liners

```python
import shap

# TreeSHAP (XGBoost, LightGBM, Random Forest, Decision Tree)
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)            # (n, p) or list for multi-class

# KernelSHAP (any model — slow)
explainer   = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:50])

# DeepSHAP (PyTorch/TF neural networks)
explainer   = shap.DeepExplainer(nn_model, torch.FloatTensor(X_train[:100]))
shap_values = explainer.shap_values(torch.FloatTensor(X_test[:20]))

# LinearSHAP (linear models with speed)
explainer   = shap.LinearExplainer(lr_model, X_train)
shap_values = explainer.shap_values(X_test)

# ── Plots ─────────────────────────────────────────────────────────────────────
shap.summary_plot(sv, X_df)                             # beeswarm (direction + magnitude)
shap.summary_plot(sv, X_df, plot_type='bar')            # global bar chart
shap.plots.waterfall(explainer(X_test)[0])              # single prediction breakdown
shap.force_plot(explainer.expected_value, sv[0], X_df.iloc[0])  # force plot
shap.dependence_plot('feature_name', sv, X_df)         # dependence + interaction
shap.plots.heatmap(explainer(X_test[:50]))              # heatmap across samples

# ── Programmatic ──────────────────────────────────────────────────────────────
mean_abs = pd.Series(np.abs(shap_values).mean(0), index=feature_names)
top5     = mean_abs.sort_values(ascending=False).head(5)

# Per-sample top feature
for i, sv_row in enumerate(shap_values[:5]):
    top_feat_idx = np.argmax(np.abs(sv_row))
    print(f"Sample {i}: top feature = {feature_names[top_feat_idx]} "
          f"(SHAP = {sv_row[top_feat_idx]:+.4f})")
```

---

## LIME One-Liners

```python
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text   import LimeTextExplainer
from lime.lime_image  import LimeImageExplainer

# Tabular
exp = LimeTabularExplainer(X_train, feature_names=feature_names,
                            class_names=['No', 'Yes'], mode='classification') \
      .explain_instance(X_test[0], model.predict_proba, num_features=10)
print(exp.as_list())                    # [(condition, weight), ...]
exp.show_in_notebook(show_table=True)
exp.save_to_file('lime_tab.html')

# Text
exp = LimeTextExplainer(class_names=['Neg', 'Pos']) \
      .explain_instance(text, pipe.predict_proba, num_features=10)
print(exp.as_list())

# Image
exp = LimeImageExplainer() \
      .explain_instance(img_array, model.predict, top_labels=1, num_samples=1000)
img_out, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True,
                                        num_features=5, hide_rest=False)
```

---

## PDP / ICE One-Liners

```python
from sklearn.inspection import PartialDependenceDisplay

# 1D PDP
PartialDependenceDisplay.from_estimator(model, X, ['feature_0'], kind='average')

# 1D ICE + PDP
PartialDependenceDisplay.from_estimator(model, X, ['feature_0'],
                                         kind='both', subsample=100, alpha=0.3)

# 2D PDP (interaction)
PartialDependenceDisplay.from_estimator(model, X, [('f0', 'f1')], kind='average')

# Multiple features at once
PartialDependenceDisplay.from_estimator(model, X, ['f0','f1','f2'], n_cols=3)
```

---

## Permutation Importance One-Liner

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=20,
                                 scoring='roc_auc', random_state=42, n_jobs=-1)
imp_df = pd.DataFrame({'feature': feature_names,
                        'mean': result.importances_mean,
                        'std':  result.importances_std}) \
           .sort_values('mean', ascending=False)
```

---

## Grad-CAM One-Liner Pattern

```python
# Hook → forward → backward → pool gradients → multiply by activations → ReLU → normalise

acts, grads = [], []
h1 = layer.register_forward_hook(lambda m,i,o: acts.append(o.detach()))
h2 = layer.register_full_backward_hook(lambda m,gi,go: grads.append(go[0].detach()))
out = model(img_t); model.zero_grad(); out[0, cls].backward()
h1.remove(); h2.remove()
cam = torch.relu((grads[0].squeeze().mean(dim=(1,2))[:,None,None] * acts[0].squeeze()).sum(0))
cam = (cam / cam.max()).numpy()
```

---

## Fairness Metrics Quick Reference

```python
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference,
    equalized_odds_difference, selection_rate
)
from sklearn.metrics import accuracy_score

# Per-group breakdown
mf = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
    y_true=y_test, y_pred=y_pred,
    sensitive_features=sensitive_feature_column
)
print(mf.by_group)
print(f"Max diff: {mf.difference(method='between_groups')}")

# Summary statistics
dp  = demographic_parity_difference(y_test, y_pred, sensitive_features=s)
eo  = equalized_odds_difference(y_test, y_pred, sensitive_features=s)
# |dp| < 0.1 and |eo| < 0.1 are common thresholds for "fair"
```

---

## Regulatory Cheatsheet

| Regulation | Scope | Key Requirement for ML |
|---|---|---|
| **GDPR Art. 22** | EU | Right to explanation for automated decisions; right to object to profiling |
| **EU AI Act (High-Risk)** | EU | Conformity assessment; human oversight; technical documentation; bias testing |
| **ECOA (USA)** | Lending | Adverse action notice must list top reasons for denial |
| **Fair Housing Act** | Housing/mortgages | No discriminatory advertising or automated underwriting |
| **HIPAA** | Healthcare | Privacy of PHI; security safeguards for ML using health data |

**Practical steps for compliance:**
1. Run MetricFrame and document per-group performance
2. Write a Model Card before deployment
3. Implement adverse action reason codes (top SHAP features)
4. Add human review for borderline predictions
5. Schedule quarterly bias audits
