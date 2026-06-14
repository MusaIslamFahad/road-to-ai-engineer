# Quick Start Guide

Choose your entry point based on your current background.

---

## 🚀 I'm a Complete Beginner

**Start here → [GETTING_STARTED.md](./GETTING_STARTED.md)**

Then follow this order:
1. `00-prerequisites` → Python basics + math
2. `01-python-for-data-science` → NumPy, Pandas
3. `02-introduction-to-ml` → Your first real ML project
4. Continue in order through the phases

**Time to first project**: ~2–3 months

---

## 💻 I'm a Software Engineer (but new to ML/AI)

You can skip basic Python but still need the math and ML fundamentals:

1. Skim `00-prerequisites` (do the math sections thoroughly)
2. `01-python-for-data-science` — Focus on NumPy/Pandas (different from web dev!)
3. `02-introduction-to-ml` → start here for ML concepts
4. Continue through Phases 2–8 at an accelerated pace

**Time saving**: ~2–3 months vs. complete beginner

---

## 📊 I'm a Data Scientist wanting to become an AI Engineer

You know classical ML well. Fast-track to the AI Engineer differentiators:

1. `09-neural-networks-basics` → Deep learning theory
2. `10-deep-learning-frameworks` → PyTorch hands-on
3. `11-computer-vision` → CV domain
4. `12-natural-language-processing` → NLP + Transformers
5. `25-generative-ai-llms` → GenAI stack
6. `13-model-deployment` + `14-mlops-basics` → Production skills
7. `22–24` → RL, GNNs, Audio (advanced specializations)

**Skip**: Modules 02–08 (you know this)
**Time saving**: ~8–12 months

---

## 🤖 I know Deep Learning but want GenAI/LLM skills

1. `12-natural-language-processing` (Transformer section)
2. `25-generative-ai-llms` → Core GenAI stack
3. `13-model-deployment` → Deploy LLM APIs
4. Then `22–24` for advanced specializations

**Time saving**: ~18 months

---

## 🎯 I want to get to Generative AI as fast as possible

**Minimum viable path to GenAI competence**:

| Module | What you get | Time |
|---|---|---|
| 00 | Python OOP + math basics | 3 weeks |
| 01 | Pandas + NumPy | 3 weeks |
| 09 | Neural network theory | 2 weeks |
| 10 | PyTorch basics | 2 weeks |
| 12 | Transformers + BERT/GPT | 4 weeks |
| 25 | RAG + Agents + LangChain | 4 weeks |

**Total**: ~4–5 months to build production GenAI apps
**Trade-off**: You'll lack breadth in CV, RL, GNNs — come back for those later

---

## 📚 Module Quick Reference

| Module | What It Covers | Start After |
|---|---|---|
| `00-prerequisites` | Python, Math, Git | Anywhere — this is the start |
| `01-python-for-data-science` | NumPy, Pandas, Visualization | Module 00 |
| `02-introduction-to-ml` | ML concepts and workflow | Module 01 |
| `03-supervised-learning-regression` | Linear regression, regularization | Module 02 |
| `04-supervised-learning-classification` | Logistic reg., SVM, trees | Module 03 |
| `05-model-evaluation-optimization` | CV, tuning, validation | Module 04 |
| `06-ensemble-methods` | Random Forest, XGBoost | Module 05 |
| `07-feature-engineering` | Encoding, scaling, pipelines | Module 06 |
| `08-unsupervised-learning` | K-Means, PCA, anomaly detection | Module 07 |
| `09-neural-networks-basics` | Neural nets, backprop, optimizers | Module 08 |
| `10-deep-learning-frameworks` | PyTorch, TensorFlow | Module 09 |
| `11-computer-vision` | CNNs, detection, GANs, diffusion | Module 10 |
| `12-natural-language-processing` | NLP, Transformers, fine-tuning | Module 10 |
| `15-time-series-analysis` | ARIMA, LSTM forecasting | Module 09 or after |
| `25-generative-ai-llms` | LLMs, RAG, agents, LangChain | Module 12 |
| `19-sql-database-fundamentals` | SQL, NoSQL, vector DBs | Module 01 or later |
| `20-handling-imbalanced-data` | SMOTE, class weights | Module 04 or later |
| `21-model-explainability` | SHAP, LIME | Module 06 or later |
| `13-model-deployment` | APIs, Docker, cloud | Module 10 or later |
| `14-mlops-basics` | MLflow, DVC, CI/CD | Module 13 |
| `22-reinforcement-learning` | MDPs, DQN, PPO | Module 10 |
| `23-graph-neural-networks` | GCNs, GATs, graph tasks | Module 10 |
| `24-audio-speech-processing` | ASR, TTS, audio ML | Module 10 |

---

## ⚡ Phase 0 in 60 Minutes (Absolute Minimum)

If you just want to verify Python is working and see ML in action:

```bash
# 1. Install
pip install scikit-learn pandas matplotlib numpy

# 2. Run this in a Python file or Jupyter cell:
```

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load classic iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
import pandas as pd
importance_df = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(importance_df.to_string(index=False))
```

You should see ~97% accuracy on the iris dataset. **Welcome to Machine Learning!**

---

## 📦 What's in `requirements.txt`?

The `requirements.txt` file includes everything for the full AI Engineer path:

- **Data**: numpy, pandas, polars, dask, scipy
- **Visualization**: matplotlib, seaborn, plotly, dash, streamlit
- **Classical ML**: scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn, optuna
- **Deep Learning**: torch, torchvision, torchaudio, tensorflow, keras
- **Computer Vision**: opencv-python, albumentations, ultralytics, timm
- **NLP & Transformers**: transformers, datasets, tokenizers, peft, accelerate, sentencepiece
- **GenAI & LLM**: langchain, langchain-openai, langchain-anthropic, openai, anthropic, chromadb, faiss-cpu, llama-index
- **Audio**: librosa, soundfile, torchaudio, speechbrain
- **RL**: gymnasium, stable-baselines3
- **GNN**: torch-geometric, dgl
- **MLOps**: mlflow, wandb, dvc, fastapi, uvicorn, docker (separate install)
- **SQL**: sqlalchemy, psycopg2-binary, pymongo
- **Explainability**: shap, lime

> **Note**: The full install is ~5 GB. For specific modules, install only what you need. GPU packages (CUDA) require a compatible GPU and separate installation.
