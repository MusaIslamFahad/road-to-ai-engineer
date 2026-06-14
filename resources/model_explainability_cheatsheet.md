# Model Explainability Cheatsheet

Quick reference for SHAP, LIME, feature importance, and interpretability tools.

---

## When to Use Which Method

| Method | Type | Model-Agnostic? | Speed | Best For |
|---|---|---|---|---|
| **Impurity importance** | Global | ❌ Tree only | Fast | Quick feature ranking |
| **Permutation importance** | Global | ✅ | Medium | Reliable feature ranking |
| **SHAP TreeExplainer** | Local + Global | ❌ Tree only | Fast | Tree models — go-to |
| **SHAP KernelExplainer** | Local | ✅ | Slow | Any model (slow) |
| **SHAP DeepExplainer** | Local | ❌ DL only | Medium | Neural networks |
| **LIME** | Local | ✅ | Medium | Quick local explanation |
| **PDP** | Global | ✅ | Medium | Marginal feature effects |
| **ICE** | Local | ✅ | Medium | Individual effect curves |
| **Grad-CAM** | Local | ❌ CNN only | Fast | Image explanations |

---

## SHAP (SHapley Additive exPlanations)

### Tree Models (XGBoost, LightGBM, Random Forest)

```python
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# Train model
model = xgb.XGBClassifier(n_estimators=200, max_depth=6)
model.fit(X_train, y_train)

# Create explainer (fast exact SHAP for tree models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)   # Shape: (n_samples, n_features)

# ─── Global: Feature Importance Bar Chart ────────────────────────────────────
shap.summary_plot(shap_values, X_test, plot_type="bar")

# ─── Global: Beeswarm Plot (direction + magnitude per sample) ────────────────
shap.summary_plot(shap_values, X_test)

# ─── Local: Waterfall Plot (single prediction explanation) ───────────────────
shap.plots.waterfall(explainer(X_test)[0])   # First test sample

# ─── Local: Force Plot ────────────────────────────────────────────────────────
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# ─── Dependence Plot (feature interaction) ────────────────────────────────────
shap.dependence_plot("tenure_months", shap_values, X_test,
                     interaction_index="monthly_charges")

# ─── Multi-class: mean |SHAP| per class ──────────────────────────────────────
# shap_values is list of arrays for multi-class
shap.summary_plot(shap_values, X_test, class_names=["Retain", "Churn"])
```

### Sklearn / Any Model (KernelSHAP — Slower)

```python
import shap
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Use a small background dataset for efficiency
background = shap.sample(X_train, 100)   # 100 random samples as reference
explainer = shap.KernelExplainer(model.predict_proba, background)

# Explain a small subset (KernelSHAP is slow)
shap_values = explainer.shap_values(X_test[:20])
shap.summary_plot(shap_values[1], X_test[:20])   # Class 1 (positive class)
```

### Neural Networks (DeepSHAP)

```python
import shap, torch

model.eval()
background = X_train[:100]   # Tensor background

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test[:10])

shap.summary_plot(shap_values, X_test[:10].numpy(), feature_names=feature_names)
```

### Programmatic SHAP Values

```python
import shap, numpy as np

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Get top 5 features for a single prediction
sample_idx = 0
feature_shap = dict(zip(X_test.columns, shap_values[sample_idx]))
top5 = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
print("Top 5 drivers for this prediction:")
for feat, val in top5:
    direction = "↑ increases" if val > 0 else "↓ decreases"
    print(f"  {feat}: {direction} churn probability by {abs(val):.4f}")

# Mean absolute SHAP (global importance)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({"feature": X_test.columns,
                               "mean_abs_shap": mean_abs_shap})\
                  .sort_values("mean_abs_shap", ascending=False)
```

---

## LIME (Local Interpretable Model-Agnostic Explanations)

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Retain", "Churn"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

# Explain a single prediction
explanation = explainer.explain_instance(
    data_row=X_test.iloc[0].values,
    predict_fn=model.predict_proba,
    num_features=10,         # Show top 10 features
    num_samples=1000         # Perturbation samples
)

# Show in notebook
explanation.show_in_notebook(show_table=True)

# Get as list of (feature_condition, weight) tuples
print(explanation.as_list())
# [('tenure_months <= 12', 0.18), ('monthly_charges > 80', 0.14), ...]

# Save as HTML
explanation.save_to_file("lime_explanation.html")
```

### LIME for Text

```python
from lime.lime_text import LimeTextExplainer

text_explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])

explanation = text_explainer.explain_instance(
    text_instance="This product is absolutely amazing and I love it!",
    classifier_fn=classifier.predict_proba,
    num_features=8
)
explanation.show_in_notebook()
```

### LIME for Images

```python
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

image_explainer = LimeImageExplainer()

explanation = image_explainer.explain_instance(
    image=image_array,    # Shape: (H, W, C), values 0–1
    classifier_fn=model.predict,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

# Get superpixel mask for top class
image_out, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)
plt.imshow(mark_boundaries(image_out, mask))
```

---

## Partial Dependence Plots (PDP)

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Single feature PDP
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=["tenure_months", "monthly_charges"],
    kind="average"        # "average" = PDP, "individual" = ICE
)
plt.tight_layout()
plt.show()

# 2D interaction PDP
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=[("tenure_months", "monthly_charges")],
    kind="average"
)
```

### ICE (Individual Conditional Expectation)

```python
# ICE shows per-sample effect lines (heterogeneity)
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=["tenure_months"],
    kind="both",              # Shows ICE lines + PDP line
    subsample=100,            # Only plot 100 samples (visual clarity)
    alpha=0.3                 # Transparency for ICE lines
)
```

---

## Permutation Feature Importance

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=30,       # Permute each feature 30 times
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": result.importances_mean,
    "importance_std":  result.importances_std
}).sort_values("importance_mean", ascending=False)

# Plot
importance_df.head(15).plot(
    kind="barh", x="feature", y="importance_mean",
    xerr="importance_std", figsize=(10, 6)
)
plt.title("Permutation Feature Importance (AUC)")
plt.tight_layout()
```

---

## Grad-CAM for CNNs

```python
import torch, cv2, numpy as np
from torchvision import transforms

def grad_cam(model, image_tensor, target_layer, class_idx=None):
    """Generate Grad-CAM heatmap for a CNN prediction."""
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    output = model(image_tensor.unsqueeze(0))

    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    grads   = gradients[0].squeeze()    # (C, H, W)
    acts    = activations[0].squeeze()  # (C, H, W)

    weights = grads.mean(dim=(1, 2))    # Global average pooling
    cam = (weights[:, None, None] * acts).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)     # Normalize to [0, 1]

    return cam.detach().numpy()

# Usage
from torchvision.models import resnet50
model = resnet50(pretrained=True)
cam = grad_cam(model, image_tensor, target_layer=model.layer4[-1])

# Overlay on original image
cam_resized = cv2.resize(cam, (image_array.shape[1], image_array.shape[0]))
heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
cv2.imwrite("gradcam_result.jpg", overlay)
```

---

## SHAP for LLMs (Text)

```python
import shap
from transformers import pipeline

# Text classification with Hugging Face
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict(texts):
    results = classifier(texts)
    return [[r["score"] if r["label"] == "POSITIVE" else 1-r["score"],
             r["score"] if r["label"] == "NEGATIVE" else 1-r["score"]]
            for r in results]

texts = ["This movie was absolutely terrible and boring.",
         "I loved every moment of this incredible film!"]

explainer = shap.Explainer(predict, shap.maskers.Text())
shap_values = explainer(texts)
shap.plots.text(shap_values[0])   # Highlight tokens by importance
```

---

## Business-Friendly Explanation Template

```python
def explain_prediction_business(model, explainer, sample, feature_names):
    """Generate a plain-English explanation for business stakeholders."""
    shap_vals = explainer.shap_values(sample.reshape(1, -1))[0]
    base_value = explainer.expected_value
    prediction = model.predict_proba(sample.reshape(1, -1))[0][1]

    top_positive = [(feature_names[i], shap_vals[i])
                    for i in range(len(shap_vals)) if shap_vals[i] > 0]
    top_positive = sorted(top_positive, key=lambda x: x[1], reverse=True)[:3]

    top_negative = [(feature_names[i], shap_vals[i])
                    for i in range(len(shap_vals)) if shap_vals[i] < 0]
    top_negative = sorted(top_negative, key=lambda x: x[1])[:3]

    print(f"Churn Probability: {prediction:.1%}")
    print(f"\nTop factors INCREASING churn risk:")
    for feat, val in top_positive:
        print(f"  • {feat}: +{val:.3f}")
    print(f"\nTop factors DECREASING churn risk:")
    for feat, val in top_negative:
        print(f"  • {feat}: {val:.3f}")

explain_prediction_business(model, explainer, X_test.iloc[5].values, feature_names)
```
