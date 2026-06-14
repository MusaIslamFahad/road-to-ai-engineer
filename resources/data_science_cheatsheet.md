# Data Science Cheatsheet

Quick-reference for the core libraries used throughout the AI Engineer path.

---

## NumPy

```python
import numpy as np

# Creation
a = np.array([1, 2, 3])
z = np.zeros((3, 4))
o = np.ones((2, 3), dtype=np.float32)
r = np.random.randn(3, 3)          # Standard normal
i = np.eye(4)                       # Identity
l = np.linspace(0, 1, 50)          # 50 evenly spaced points
ar = np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]

# Shape manipulation
a.reshape(3, 1)                     # New shape (no copy if possible)
a.ravel()                           # Flatten to 1D
np.expand_dims(a, axis=0)           # Add dimension
np.squeeze(a)                       # Remove size-1 dimensions

# Indexing
a[1:3]                              # Slice
a[[0, 2]]                           # Fancy index
a[a > 5]                            # Boolean mask
np.where(a > 5, a, 0)              # Conditional selection

# Math & Stats
np.dot(A, B)                        # Matrix multiply (2D)
A @ B                               # Same as dot for 2D
np.linalg.inv(A)                    # Matrix inverse
np.linalg.eig(A)                    # Eigenvalues and eigenvectors
U, S, Vt = np.linalg.svd(A)        # SVD
np.mean(a, axis=0)                  # Mean along axis
np.std(a, ddof=1)                   # Sample standard deviation
np.corrcoef(x, y)                   # Correlation matrix

# Broadcasting example
a = np.ones((3, 1))
b = np.ones((1, 4))
(a + b).shape                       # (3, 4) — broadcasts
```

---

## Pandas

```python
import pandas as pd

# Load
df = pd.read_csv('data.csv', parse_dates=['date'])
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_json('data.json')
df = pd.read_sql(query, con=engine)

# Inspect
df.head(5)
df.info()
df.describe(include='all')
df.shape, df.dtypes, df.columns

# Indexing
df.loc[0:5, ['col1', 'col2']]       # Label-based
df.iloc[0:5, 0:2]                   # Position-based
df.query("age > 30 and income > 50000")

# Cleaning
df.isnull().sum()
df.fillna(df.median(numeric_only=True))
df.dropna(subset=['target'])
df.drop_duplicates(subset=['id'])
df['col'] = df['col'].astype('float32')
df.rename(columns={'old': 'new'}, inplace=True)

# Feature Engineering
df['year'] = df['date'].dt.year
df['log_price'] = np.log1p(df['price'])
df['is_expensive'] = (df['price'] > 1000).astype(int)

# GroupBy
(df.groupby('category')['revenue']
   .agg(['mean', 'sum', 'count'])
   .reset_index()
   .sort_values('sum', ascending=False))

# Merge / Join
merged = pd.merge(df1, df2, on='key', how='left')
concat = pd.concat([df1, df2], ignore_index=True)

# Pivot
pivot = df.pivot_table(values='sales', index='region', columns='product', aggfunc='sum')

# Apply
df['label'] = df['score'].apply(lambda x: 'High' if x > 0.5 else 'Low')
df['normalized'] = df.groupby('group')['value'].transform(lambda x: (x - x.mean()) / x.std())

# Save
df.to_csv('output.csv', index=False)
df.to_parquet('output.parquet', engine='pyarrow')
```

---

## Scikit-learn

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline
num_pipe = Pipeline([('scaler', StandardScaler())])
cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
preproc = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
clf = Pipeline([('preproc', preproc), ('model', RandomForestClassifier())])
clf.fit(X_train, y_train)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# Hyperparameter tuning with Optuna
import optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Tensor basics
x = torch.tensor([1.0, 2.0], requires_grad=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)

# Custom model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(128, 256, 10).to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(loader), epochs=10)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

# Save / Load
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt', map_location=device))
```

---

## FastAPI

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import joblib, numpy as np

app = FastAPI(title="ML API", version="1.0")
model = joblib.load("model.pkl")

class InputFeatures(BaseModel):
    feature1: float = Field(..., ge=0, le=1000, description="Feature 1")
    feature2: float
    feature3: int = Field(..., ge=0, le=50)

class PredictionResponse(BaseModel):
    prediction: float
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: InputFeatures):
    X = np.array([[data.feature1, data.feature2, data.feature3]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return PredictionResponse(prediction=float(pred), probability=float(prob))

@app.get("/health")
async def health():
    return {"status": "ok"}

# Run: uvicorn app:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

## OpenCV

```python
import cv2
import numpy as np

# Read / Show / Write
img = cv2.imread('image.jpg')           # BGR format
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.jpg', img)

# Resize / Crop
resized = cv2.resize(img, (224, 224))
cropped = img[y:y+h, x:x+w]

# Color spaces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Drawing
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.circle(img, (cx, cy), radius, (255, 0, 0), -1)
cv2.putText(img, 'Label', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# Normalize for DL
img_norm = img.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
img_norm = (img_norm - mean) / std
img_tensor = img_norm.transpose(2, 0, 1)   # HWC → CHW
```

---

## Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load model & tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

dataset = load_dataset('imdb')
tokenized = dataset.map(tokenize, batched=True)

# Train
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    learning_rate=2e-5,
    weight_decay=0.01,
)
trainer = Trainer(model=model, args=args, train_dataset=tokenized['train'], eval_dataset=tokenized['test'])
trainer.train()

# Inference pipeline
from transformers import pipeline
classifier = pipeline("text-classification", model="./results/best_model")
result = classifier("This movie was fantastic!")
```

---

## LangChain (LCEL)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# Simple chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("Answer in one sentence: {question}")
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"question": "What is RAG?"}))

# RAG chain
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
Context: {context}
Question: {question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)
print(rag_chain.invoke("What does the document say about X?"))
```

---

## MLflow

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("house-price-experiment")

with mlflow.start_run(run_name="ridge-regression-v1"):
    # Log params
    mlflow.log_params({"alpha": 1.0, "normalize": True})

    # Train
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Log metrics
    preds = model.predict(X_test)
    mlflow.log_metrics({"rmse": rmse(y_test, preds), "r2": r2(y_test, preds)})

    # Log model
    mlflow.sklearn.log_model(model, "ridge-model")
    mlflow.log_artifact("feature_importance.png")
```

---

## Key Evaluation Metrics Quick Reference

| Metric | Formula | Use When |
|---|---|---|
| Accuracy | TP+TN / Total | Balanced classes |
| Precision | TP / (TP+FP) | FP is costly (spam filter) |
| Recall | TP / (TP+FN) | FN is costly (cancer detection) |
| F1 | 2 × P × R / (P+R) | Imbalanced, both matter |
| ROC-AUC | Area under ROC | General classifier ranking |
| PR-AUC | Area under PR | Highly imbalanced datasets |
| RMSE | √(mean((y-ŷ)²)) | Regression, penalize outliers |
| MAE | mean(|y-ŷ|) | Regression, robust to outliers |
| R² | 1 - SS_res/SS_tot | Regression, % variance explained |
| MAPE | mean(|y-ŷ|/y)×100 | Relative error (forecasting) |
