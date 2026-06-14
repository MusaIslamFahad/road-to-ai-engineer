# Getting Started: Your First 30 Minutes as an AI Engineer

Welcome! This guide gets you from zero to running your first ML model in 30 minutes. No prior experience needed.

---

## Step 1: Install Python (5 minutes)

### Option A: Anaconda (Recommended — includes everything)
1. Go to [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)
2. Download the Python 3.10+ installer for your OS
3. Run the installer (accept default options)
4. Open **Anaconda Navigator** or **Anaconda Prompt**

### Option B: Python + pip (Minimal)
1. Download Python 3.10+ from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. During installation, check **"Add Python to PATH"**
3. Open a terminal and verify: `python --version`

---

## Step 2: Clone This Repository (2 minutes)

```bash
git clone https://github.com/YOUR_USERNAME/road-to-ai-engineer.git
cd road-to-ai-engineer
```

If you don't have Git: [Download Git](https://git-scm.com/downloads)

---

## Step 3: Create Your Environment (3 minutes)

### With Anaconda:
```bash
conda create -n ai-engineer python=3.10
conda activate ai-engineer
pip install -r requirements.txt
```

### With Python venv:
```bash
python -m venv ai-engineer
# Windows:
ai-engineer\Scripts\activate
# Mac/Linux:
source ai-engineer/bin/activate

pip install -r requirements.txt
```

---

## Step 4: Open Jupyter Notebook (2 minutes)

```bash
pip install jupyter notebook
jupyter notebook
```

Your browser will open at `http://localhost:8888`. Navigate to the `00-prerequisites` folder.

---

## Step 5: Your First ML Model (18 minutes)

Create a new notebook and run this code cell by cell:

```python
# Cell 1: Install and import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
print("All imports successful! ✅")
```

```python
# Cell 2: Create sample data
np.random.seed(42)
n = 200
# House features: size (sq ft) and age (years)
house_size = np.random.normal(1500, 500, n)
house_age  = np.random.uniform(0, 50, n)
# Price depends on size and age (plus random noise)
price = 100 * house_size - 2000 * house_age + np.random.normal(0, 20000, n)

df = pd.DataFrame({
    'size':  house_size,
    'age':   house_age,
    'price': price
})
print(df.head())
print(f"\nDataset shape: {df.shape}")
```

```python
# Cell 3: Explore the data
print("Summary Statistics:")
print(df.describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(df['size'],  df['price'], alpha=0.5, color='blue')
axes[0].set_xlabel('House Size (sq ft)')
axes[0].set_ylabel('Price ($)')
axes[0].set_title('Price vs Size')

axes[1].scatter(df['age'], df['price'], alpha=0.5, color='orange')
axes[1].set_xlabel('House Age (years)')
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Price vs Age')

plt.tight_layout()
plt.show()
```

```python
# Cell 4: Split data
X = df[['size', 'age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set:   {X_train.shape[0]} samples")
print(f"Test set:       {X_test.shape[0]} samples")
```

```python
# Cell 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"  Intercept: {model.intercept_:.2f}")
```

```python
# Cell 6: Evaluate
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"RMSE:  ${rmse:,.0f}")
print(f"R²:    {r2:.4f} ({r2*100:.1f}% of variance explained)")

# Visualize predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.tight_layout()
plt.show()
```

```python
# Cell 7: Make a prediction
new_house = pd.DataFrame({'size': [2000], 'age': [10]})
predicted_price = model.predict(new_house)[0]
print(f"A 2000 sq ft house that's 10 years old:")
print(f"Predicted price: ${predicted_price:,.0f}")
```

🎉 **Congratulations!** You just trained your first ML model!

---

## What You Just Did

1. **Created data** with known relationships (price depends on size and age)
2. **Explored** the data visually with scatter plots
3. **Split** data into training and test sets (avoiding data leakage)
4. **Trained** a Linear Regression model to learn the price formula
5. **Evaluated** the model using RMSE and R² metrics
6. **Made predictions** on a new house

---

## What's Next?

### If this felt easy:
→ Head directly to [Module 02: Introduction to ML](./02-introduction-to-ml/README.md)

### If you want to understand the math first:
→ Start with [Module 00: Prerequisites](./00-prerequisites/README.md)

### If you want to jump to deep learning right away:
→ You need Phases 0–5 first — don't skip! The foundations make everything easier.

---

## Free Cloud Alternatives (No Install Needed)

If you can't install software locally:

| Platform | What it offers | Link |
|---|---|---|
| **Google Colab** | Free GPU, Jupyter notebooks | [colab.research.google.com](https://colab.research.google.com) |
| **Kaggle Kernels** | Free GPU + datasets | [kaggle.com](https://www.kaggle.com) |
| **GitHub Codespaces** | Full dev environment | [github.com/codespaces](https://github.com/codespaces) |

For the deep learning modules (09–12), you'll want a GPU. Colab and Kaggle both offer free T4/P100 GPUs.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'sklearn'`**
```bash
pip install scikit-learn
```

**`conda: command not found`**
- Close and reopen your terminal after Anaconda installation
- Or add Anaconda to your PATH manually

**`jupyter: command not found`**
```bash
pip install jupyter notebook
```

**Port 8888 already in use**
```bash
jupyter notebook --port 8889
```

**ImportError on Windows with numpy**
```bash
pip install --upgrade numpy
```

---

## Recommended Setup for Deep Learning (Phase 5+)

For Modules 09–25, you'll benefit from:

```bash
# PyTorch with CUDA (GPU support) — check pytorch.org for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorFlow
pip install tensorflow

# Hugging Face
pip install transformers datasets accelerate peft

# LLM / GenAI stack
pip install langchain langchain-openai langchain-anthropic chromadb openai

# Computer Vision
pip install opencv-python albumentations ultralytics  # for YOLO

# MLOps
pip install mlflow wandb dvc
```

Or just run `pip install -r requirements.txt` from the root of this repository — it includes everything.

---

Ready to begin the full journey? Start with [Module 00: Prerequisites →](./00-prerequisites/README.md)
