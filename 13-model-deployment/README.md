# Module 13: Model Deployment

**Phase 8 — Production & Deployment** | Est. time: 1.5–2 months (full-time) · 3–4 months (part-time)

---

## Learning Objectives

By the end of this module you will:
- Build production REST APIs with FastAPI (validation, async, streaming, auth)
- Containerise ML models with Docker (multi-stage builds, GPU, docker-compose)
- Deploy to AWS SageMaker, GCP Vertex AI, and Hugging Face Spaces
- Design and run statistically valid A/B tests
- Monitor models for data drift and performance degradation in production

---

## Prerequisites

- Any trained model (Modules 02–12); basic command-line familiarity

---

## 1. Model Serialisation

```python
# ─── sklearn / XGBoost / LightGBM ────────────────────────────────────────────
import joblib
joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")

# ─── PyTorch state_dict (preferred for training/fine-tuning) ─────────────────
import torch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# ─── ONNX (cross-platform, fast inference) ───────────────────────────────────
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}}, opset_version=17)

import onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
output = sess.run(None, {"input": X_np})[0]

# ─── Hugging Face Hub ────────────────────────────────────────────────────────
model.save_pretrained("./my_model"); tokenizer.save_pretrained("./my_model")
model.push_to_hub("your-username/my-model")
```

---

## 2. FastAPI Production API

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib, numpy as np, time, logging

logger = logging.getLogger(__name__)
app    = FastAPI(title="Churn Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                  allow_methods=["*"], allow_headers=["*"])

# Load model once at startup
model = None
@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model.pkl")
    logger.info("Model loaded ✅")

class ChurnRequest(BaseModel):
    age:            float = Field(..., ge=18, le=100)
    tenure_months:  int   = Field(..., ge=0, le=360)
    monthly_charges:float = Field(..., ge=0)
    contract_type:  str   = Field(..., pattern="^(monthly|annual|biennial)$")

    @validator("monthly_charges")
    def reasonable_charge(cls, v):
        if v > 10_000: raise ValueError("Charge unreasonably high")
        return v

class ChurnResponse(BaseModel):
    churn_probability: float
    prediction:        str     # "churn" | "retain"
    confidence:        str     # "high" | "medium" | "low"
    model_version:     str = "1.0.0"

def _log_prediction(data: dict, result: dict):
    logger.info(f"prediction | input={data} | output={result}")

@app.post("/predict", response_model=ChurnResponse)
async def predict(req: ChurnRequest, bg: BackgroundTasks):
    t0 = time.perf_counter()
    contract_enc = {"monthly": 0, "annual": 1, "biennial": 2}[req.contract_type]
    X = np.array([[req.age, req.tenure_months, req.monthly_charges, contract_enc]])
    prob = float(model.predict_proba(X)[0][1])
    pred = "churn" if prob > 0.5 else "retain"
    conf = ("high" if abs(prob - 0.5) > 0.3 else
            "medium" if abs(prob - 0.5) > 0.1 else "low")
    result = ChurnResponse(churn_probability=round(prob, 4),
                           prediction=pred, confidence=conf)
    bg.add_task(_log_prediction, req.dict(), result.dict())
    logger.info(f"Inference: {(time.perf_counter()-t0)*1000:.1f}ms")
    return result

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
```

---

## 3. Docker

### Dockerfile (production-grade)
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first — layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for security
RUN adduser --disabled-password appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### docker-compose.yml
```yaml
version: "3.8"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - MODEL_PATH=/app/model.pkl
    volumes:
      - ./model.pkl:/app/model.pkl:ro
    restart: unless-stopped
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

```bash
docker build -t churn-api:1.0 .
docker run -d -p 8000:8000 --name api churn-api:1.0
docker logs -f api
docker exec -it api bash
```

---

## 4. Cloud Deployment

### AWS SageMaker
```python
import sagemaker
from sagemaker.sklearn import SKLearnModel

sm_model = SKLearnModel(
    model_data="s3://my-bucket/model.tar.gz",
    role=sagemaker.get_execution_role(),
    framework_version="1.2-1",
    entry_point="inference.py"
)
predictor = sm_model.deploy(initial_instance_count=1,
                            instance_type="ml.m5.xlarge",
                            endpoint_name="churn-v1")
result = predictor.predict([[35, 12, 65.5, 1]])
```

### Hugging Face Spaces (free GPU)
Create `app.py` with Gradio:
```python
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-classification", model="./model")
demo = gr.Interface(fn=lambda text: pipe(text)[0]["label"],
                    inputs="text", outputs="text")
demo.launch()
```
Push to a Space repo — HF handles hosting automatically.

---

## 5. A/B Testing

```python
import hashlib
from scipy import stats

def get_variant(user_id: str, split: float = 0.2) -> str:
    """Deterministic, user-sticky variant assignment."""
    bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    return "model_b" if bucket < split * 100 else "model_a"

def significance_test(conv_a, n_a, conv_b, n_b, alpha=0.05):
    rate_a, rate_b = conv_a/n_a, conv_b/n_b
    pooled = (conv_a+conv_b)/(n_a+n_b)
    se = (pooled*(1-pooled)*(1/n_a+1/n_b))**0.5
    z  = (rate_b - rate_a) / se
    p  = 2*(1 - stats.norm.cdf(abs(z)))
    return {"significant": p < alpha, "p_value": round(p, 4),
            "lift": f"{(rate_b-rate_a)/rate_a*100:+.1f}%"}
```

---

## 6. Model Monitoring

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping

col_map = ColumnMapping(
    target="churn", prediction="prediction",
    numerical_features=["age","tenure_months","monthly_charges"]
)
report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
report.run(reference_data=training_df, current_data=production_df,
           column_mapping=col_map)
report.save_html("monitoring.html")

# Programmatic check
if report.as_dict()["metrics"][0]["result"]["dataset_drift"]:
    print("⚠️ Drift detected — trigger retraining")
```

---

## Related Resources

- [Model Deployment Cheatsheet](../resources/model_deployment_cheatsheet.md)
- [Docker Tutorial](../resources/docker_tutorial.md)
- [MLflow Guide](../resources/mlflow_comprehensive_guide.md)

**[← Module 21](../21-model-explainability/README.md)** | **[→ Module 14](../14-mlops-basics/README.md)**
