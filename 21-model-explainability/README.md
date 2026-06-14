# Module 21: Model Explainability

**Phase 7.5 — Essential Skills** | Est. time: 3–4 weeks (full-time) · 6–8 weeks (part-time)

*Best taken after Module 06 (Ensemble Methods)*

## Topics Covered

### Global Interpretability
- **Impurity-based feature importance**: from tree/forest models; biased toward high-cardinality features
- **Permutation importance**: model-agnostic; shuffle feature and measure accuracy drop
- **Partial Dependence Plots (PDP)**: marginal effect of one or two features on prediction
- **Individual Conditional Expectation (ICE)**: PDP per data point — reveals heterogeneous effects

### Local Interpretability
- **SHAP (SHapley Additive exPlanations)**:
  - `TreeExplainer`: exact SHAP for tree models (fast)
  - `LinearExplainer`: for linear models
  - `KernelExplainer`: model-agnostic; uses a weighted linear regression (slow)
  - `DeepExplainer`: for deep learning with PyTorch/TF
  - Plots: waterfall, force, beeswarm, bar, heatmap, scatter
- **LIME**: train a local linear model around a prediction; works for tabular, text, image

### Deep Learning Explainability
- **Grad-CAM**: gradient-weighted class activation maps for CNNs; highlight important image regions
- **Integrated Gradients**: attribution method for any differentiable model
- **SHAP DeepLIFT**: efficient approximation for neural networks

### Regulatory & Business Context
- **EU AI Act**: high-risk systems require explainability and human oversight
- **GDPR Article 22**: right to explanation for automated decisions
- **Fair lending (ECOA/FHAct)**: adverse action notices in credit decisions
- Explainability reports for business stakeholders (non-technical audience)

## Files

```
21-model-explainability/
├── README.md
├── model-explainability.md
├── model-explainability-advanced-topics.md
├── model-explainability-quick-reference.md
└── model-explainability-project-tutorial.md
```

**[← Main README](../README.md)**
