# Model Explainability — Main Guide

Complete learning material for SHAP, LIME, PDP, ICE, Grad-CAM, and Responsible AI.

---

## Why Explainability Matters

| Context | Why It's Required |
|---|---|
| **Regulatory** | EU AI Act, GDPR Art. 22 (right to explanation), ECOA (adverse action notices) |
| **Business** | Stakeholders need to trust and validate model decisions |
| **Debugging** | Understand why a model fails on specific segments |
| **Fairness** | Detect if model relies on protected attributes or proxies |
| **Science** | Gain domain insights from what the model learned |

---

## Part 1: Feature Importance Basics

### Impurity-Based (Tree Models)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Train a model
X, y = make_classification(n_samples=5000, n_features=15,
                            n_informative=8, n_redundant=3,
                            random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Impurity-based importance (built in)
importance_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

importance_df.head(10).plot(
    kind='barh', x='feature', y='importance',
    title='Feature Importance (Impurity-Based)',
    figsize=(10, 6)
)
plt.tight_layout()
plt.savefig('impurity_importance.png', dpi=150)
```

⚠️ **Caveat**: Impurity-based importance is **biased toward high-cardinality features** (continuous or many-valued). Always verify with permutation importance.

### Permutation Importance (Model-Agnostic)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature':   feature_names,
    'mean':      result.importances_mean,
    'std':       result.importances_std
}).sort_values('mean', ascending=False)

print(perm_df.head(10))

# Compare impurity vs permutation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
importance_df.head(10).plot(kind='barh', x='feature', y='importance',
                             ax=ax1, title='Impurity Importance')
perm_df.head(10).plot(kind='barh', x='feature', y='mean',
                      xerr='std', ax=ax2, title='Permutation Importance (AUC)')
plt.tight_layout()
plt.savefig('importance_comparison.png', dpi=150)
```

---

## Part 2: SHAP Values (SHapley Additive exPlanations)

SHAP is the gold standard for explainability. It assigns each feature a contribution value for each individual prediction, grounded in cooperative game theory.

**Key property**: SHAP values sum to the difference between the prediction and the baseline (expected value).

### TreeSHAP — Fast & Exact for Tree Models

```python
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Train XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Create TreeExplainer
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)   # shape: (n_samples, n_features)

print(f"Base value (expected output): {explainer.expected_value:.4f}")
print(f"SHAP values shape: {shap_values.shape}")

# ── Global: Beeswarm plot ─────────────────────────────────────────────────────
# Shows distribution of SHAP values for each feature — direction + magnitude
shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names))

# ── Global: Mean |SHAP| bar chart ─────────────────────────────────────────────
shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names),
                  plot_type='bar')

# ── Local: Waterfall plot for a single prediction ─────────────────────────────
# Shows how each feature pushed the prediction up or down from the base value
explanation = explainer(X_test)
shap.plots.waterfall(explanation[0])     # first test sample

# ── Local: Force plot ─────────────────────────────────────────────────────────
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    pd.DataFrame(X_test, columns=feature_names).iloc[0]
)

# ── Feature interaction: Dependence plot ──────────────────────────────────────
# Shows how SHAP value of one feature varies with its value, and with another feature
shap.dependence_plot('feature_0', shap_values,
                     pd.DataFrame(X_test, columns=feature_names),
                     interaction_index='feature_1')
```

### Multi-Class SHAP

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

explainer   = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
# shap_values is a LIST of arrays, one per class

# Show for class 1 (positive class in binary)
shap.summary_plot(shap_values[1], pd.DataFrame(X_test, columns=feature_names))

# Mean |SHAP| per class
shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names),
                  class_names=['Class 0', 'Class 1'], plot_type='bar')
```

### KernelSHAP — Any Model (Slower)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
pipe.fit(X_train, y_train)

# Use 100 background samples for efficiency
background  = shap.sample(X_train, 100)
explainer   = shap.KernelExplainer(pipe.predict_proba, background)
shap_values = explainer.shap_values(X_test[:50])   # KernelSHAP is slow — use small subset
print("KernelSHAP done")
shap.summary_plot(shap_values[1], pd.DataFrame(X_test[:50], columns=feature_names))
```

### SHAP for Neural Networks

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

nn_model = MLP(X_train.shape[1], 64, 2)
# ... train nn_model ...

# DeepSHAP for PyTorch
background_t = torch.FloatTensor(X_train[:100])
test_t       = torch.FloatTensor(X_test[:20])

explainer   = shap.DeepExplainer(nn_model, background_t)
shap_values = explainer.shap_values(test_t)
shap.summary_plot(shap_values[1], X_test[:20], feature_names=feature_names)
```

### Extracting SHAP Values Programmatically

```python
# Get top N features by importance for a specific prediction
def explain_prediction(model, explainer, X_sample, feature_names, n=5):
    """Return the top N features driving a prediction."""
    shap_vals = explainer.shap_values(X_sample.reshape(1, -1))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class for binary

    contributions = dict(zip(feature_names, shap_vals[0]))
    top = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

    print(f"Prediction: {model.predict_proba(X_sample.reshape(1,-1))[0][1]:.3f}")
    print(f"Base value: {explainer.expected_value:.3f}")
    print("\nTop feature contributions:")
    for feat, val in top:
        direction = "↑ increases" if val > 0 else "↓ decreases"
        print(f"  {feat:20s}: {direction} risk by {abs(val):.4f}")

explain_prediction(model, explainer, X_test[0], feature_names)
```

---

## Part 3: LIME (Local Interpretable Model-Agnostic Explanations)

LIME fits a local linear model around a single prediction to approximate the black-box model locally.

### Tabular

```python
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Create explainer from training data
lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=['No Churn', 'Churn'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

# Explain a single prediction
idx = 0   # which test sample to explain
exp = lime_explainer.explain_instance(
    data_row=X_test[idx],
    predict_fn=model.predict_proba,
    num_features=10,
    num_samples=2000
)

# Show as list: (condition, weight) pairs
print(exp.as_list())
# [('feature_3 > 0.50', 0.18), ('feature_0 <= -0.20', -0.14), ...]

# Display in notebook
exp.show_in_notebook(show_table=True)

# Save to HTML
exp.save_to_file('lime_explanation.html')

# Programmatic access
weights = dict(exp.as_list())
most_important = max(weights, key=lambda k: abs(weights[k]))
print(f"Most important factor: {most_important} (weight: {weights[most_important]:.4f})")
```

### LIME for Text

```python
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Train text classifier
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf',   LogisticRegression(max_iter=1000))
])
text_pipeline.fit(X_text_train, y_text_train)

text_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

exp = text_explainer.explain_instance(
    text_instance="This product is absolutely amazing! I love it.",
    classifier_fn=text_pipeline.predict_proba,
    num_features=10
)
exp.show_in_notebook()
print(exp.as_list())
# [('amazing', 0.31), ('love', 0.24), ('absolutely', 0.18), ...]
```

### LIME for Images

```python
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import numpy as np

image_explainer = LimeImageExplainer()

exp = image_explainer.explain_instance(
    image=img_array,         # numpy array (H, W, C), values 0-1
    classifier_fn=model.predict,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

# Get image with highlighted superpixels
top_label = exp.top_labels[0]
temp, mask = exp.get_image_and_mask(
    top_label,
    positive_only=True,
    num_features=5,
    hide_rest=False
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(img_array);                plt.title('Original')
plt.subplot(1, 3, 2); plt.imshow(mark_boundaries(temp, mask)); plt.title('LIME Regions')
plt.subplot(1, 3, 3); plt.imshow(mask, cmap='RdYlGn');      plt.title('Positive Mask')
plt.tight_layout(); plt.savefig('lime_image.png', dpi=150)
```

---

## Part 4: Partial Dependence Plots (PDP) & ICE

PDPs show the **marginal effect** of a feature on the prediction (averaged over all other features).
ICE plots show this per-sample (heterogeneous effects).

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# ── 1D PDP: how does feature_0 affect the prediction? ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
PartialDependenceDisplay.from_estimator(
    model, X_test, features=['feature_0'],
    kind='average',        # 'average' = PDP, 'individual' = ICE, 'both' = both
    ax=ax
)
ax.set_title('Partial Dependence Plot: feature_0')
plt.tight_layout(); plt.savefig('pdp_1d.png', dpi=150)

# ── ICE + PDP together ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_test, features=['feature_0'],
    kind='both',          # shows ICE lines + bold PDP line
    subsample=100,        # plot 100 ICE lines (avoid clutter)
    alpha=0.3,
    ax=ax
)
ax.set_title('ICE + PDP: feature_0')
plt.savefig('ice_pdp.png', dpi=150)

# ── 2D PDP: interaction between two features ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    model, X_test,
    features=[('feature_0', 'feature_1')],   # tuple = 2D PDP
    kind='average',
    ax=ax
)
ax.set_title('2D PDP: feature_0 vs feature_1 Interaction')
plt.savefig('pdp_2d.png', dpi=150)

# ── Multiple PDPs at once ─────────────────────────────────────────────────────
top_features = importance_df['feature'].head(6).tolist()
PartialDependenceDisplay.from_estimator(
    model, X_test, features=top_features,
    kind='average', n_cols=3, grid_resolution=50
)
plt.suptitle('PDPs for Top 6 Features')
plt.tight_layout(); plt.savefig('pdp_grid.png', dpi=150)
```

---

## Part 5: Grad-CAM for CNNs

Grad-CAM highlights which image regions were most important for a CNN's prediction.

```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2
import matplotlib.pyplot as plt

def grad_cam(model, image_tensor, target_layer, class_idx=None):
    """
    Generate Grad-CAM heatmap.
    Args:
        model: trained CNN (PyTorch)
        image_tensor: (1, C, H, W) tensor
        target_layer: the convolutional layer to hook
        class_idx: which output class to explain (None = argmax)
    """
    activations, gradients = [], []

    def fwd_hook(module, input, output):
        activations.append(output.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.eval()
    output = model(image_tensor)

    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    h1.remove(); h2.remove()

    grads = gradients[0].squeeze()      # (C, H, W)
    acts  = activations[0].squeeze()    # (C, H, W)
    weights = grads.mean(dim=(1, 2))    # global average pooling

    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)     # normalise to [0, 1]
    return cam.numpy(), class_idx

# Usage with ResNet-50
model   = resnet50(weights=ResNet50_Weights.DEFAULT)
preproc = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# Load your image
from PIL import Image
img = Image.open('dog.jpg').convert('RGB')
img_t = preproc(img).unsqueeze(0)

# Generate Grad-CAM for last residual block
cam, pred_class = grad_cam(model, img_t, target_layer=model.layer4[-1])

# Overlay on original image
img_np    = np.array(img.resize((224, 224)))
cam_resized = cv2.resize(cam, (224, 224))
heatmap   = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay   = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(img_np);         axes[0].set_title('Original Image')
axes[1].imshow(cam, cmap='jet');axes[1].set_title(f'Grad-CAM (class {pred_class})')
axes[2].imshow(overlay);        axes[2].set_title('Overlay')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.savefig('gradcam.png', dpi=150, bbox_inches='tight')
print(f"Predicted class: {pred_class}")
```

---

## Part 6: SHAP for Text (LLMs & Transformers)

```python
import shap
from transformers import pipeline

# Load a Hugging Face text classifier
classifier = pipeline('text-classification',
                      model='distilbert-base-uncased-finetuned-sst-2-english',
                      return_all_scores=True)

def predict_proba(texts):
    results = classifier(list(texts))
    return np.array([[r['score'] for r in res] for res in results])

texts = [
    "This product is absolutely amazing!",
    "Terrible quality, very disappointed.",
    "It's okay, nothing special."
]

explainer   = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer=r'\W+'))
shap_values = explainer(texts)

# Visualise token importance
shap.plots.text(shap_values[0])    # first text example
shap.plots.bar(shap_values.abs.mean(0))  # mean importance across samples
```
