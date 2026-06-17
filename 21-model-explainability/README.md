# Module 21: Model Explainability

**Phase 7.5 — Essential Skills** | Est. time: 3–4 weeks (full-time) · 6–8 weeks (part-time)

*Best taken after Module 06 (Ensemble Methods). Can be studied alongside any later module.*

---

## Learning Objectives

By the end of this module you will:
- Compute global and local SHAP values for tree, linear, and deep learning models
- Apply LIME to explain individual predictions for tabular, text, and image data
- Create Partial Dependence Plots and ICE plots
- Generate Grad-CAM heatmaps for CNN image explanations
- Audit models for fairness using Fairlearn
- Write regulatory-grade Model Cards

---

## Prerequisites

- Module 06: Ensemble Methods (tree models); Module 10: Deep Learning (for neural net explanations)

---

## Files in This Module

```
21-model-explainability/
├── README.md                                   ← You are here
├── model-explainability.md                     ← Main guide: SHAP, LIME, PDP, Grad-CAM (with full code)
├── model-explainability-advanced-topics.md     ← Integrated Gradients, interaction values, global surrogate, Anchors
├── model-explainability-quick-reference.md     ← All method one-liners for fast lookup
└── model-explainability-project-tutorial.md    ← End-to-end project: credit risk dashboard
```

---

## Quick Start

```python
import shap
import numpy as np
import pandas as pd

# ── SHAP for any tree model (XGBoost, LightGBM, Random Forest) ───────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)   # shape: (n_samples, n_features)

# Global: which features matter most overall?
shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names))

# Local: why did the model predict THIS for customer 0?
explanation = explainer(X_test)
shap.plots.waterfall(explanation[0])

# ── LIME for a single prediction ──────────────────────────────────────────────
from lime.lime_tabular import LimeTabularExplainer
lime_exp = LimeTabularExplainer(X_train, feature_names=feature_names,
                                 class_names=['No', 'Yes'], mode='classification')
exp = lime_exp.explain_instance(X_test[0], model.predict_proba, num_features=8)
print(exp.as_list())   # [(condition, LIME_weight), ...]
```

---

## Topics Covered

| Topic | File |
|---|---|
| Feature importance (impurity + permutation) | [model-explainability.md](./model-explainability.md) |
| TreeSHAP, KernelSHAP, DeepSHAP, LinearSHAP | [model-explainability.md](./model-explainability.md) |
| SHAP plots: beeswarm, waterfall, force, dependence | [model-explainability.md](./model-explainability.md) |
| LIME: tabular, text, image | [model-explainability.md](./model-explainability.md) |
| Partial Dependence Plots (1D, 2D, ICE) | [model-explainability.md](./model-explainability.md) |
| Grad-CAM for CNNs | [model-explainability.md](./model-explainability.md) |
| SHAP for LLMs / transformers | [model-explainability.md](./model-explainability.md) |
| Integrated Gradients | [model-explainability-advanced-topics.md](./model-explainability-advanced-topics.md) |
| SHAP interaction values | [model-explainability-advanced-topics.md](./model-explainability-advanced-topics.md) |
| Global surrogate models | [model-explainability-advanced-topics.md](./model-explainability-advanced-topics.md) |
| Anchors (rule-based) | [model-explainability-advanced-topics.md](./model-explainability-advanced-topics.md) |
| Fairness-aware explainability | [model-explainability-advanced-topics.md](./model-explainability-advanced-topics.md) |
| Model Cards | [model-explainability-advanced-topics.md](./model-explainability-advanced-topics.md) |
| All syntax at a glance | [model-explainability-quick-reference.md](./model-explainability-quick-reference.md) |
| Credit risk dashboard project | [model-explainability-project-tutorial.md](./model-explainability-project-tutorial.md) |

---

## Related Resources

- [Model Explainability Cheatsheet](../resources/model_explainability_cheatsheet.md)
- [Ethics in ML](../resources/ethics_in_ml.md) — Fairness, EU AI Act, Model Cards

**[← Module 20](../20-handling-imbalanced-data/README.md)** | **[→ Module 13](../13-model-deployment/README.md)**
