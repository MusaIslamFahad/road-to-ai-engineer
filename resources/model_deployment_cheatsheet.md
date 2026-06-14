# Model Deployment Cheatsheet

Quick-reference for all aspects of deploying ML models: APIs, Docker, cloud platforms, monitoring.

---

## FastAPI Quick Reference

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import joblib, numpy as np, time, logging

logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model API", version="2.0.0", docs_url="/docs")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model once at startup
model = None
@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully")

# Input validation with Pydantic
class PredictionRequest(BaseModel):
    age:            float = Field(..., ge=18, le=100, description="Age in years")
    income:         float = Field(..., ge=0, description="Annual income in USD")
    tenure_months:  int   = Field(..., ge=0, le=360)
    contract_type:  str   = Field(..., pattern="^(monthly|annual|biennial)$")

    @validator("income")
    def income_reasonable(cls, v):
        if v > 10_000_000:
            raise ValueError("Income seems unreasonably high")
        return v

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction:        str    # "churn" | "retain"
    confidence:        str    # "high" | "medium" | "low"
    model_version:     str = "2.0.0"

def log_prediction(request: dict, response: dict):
    logger.info(f"prediction | input={request} | output={response}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    X = np.array([[req.age, req.income, req.tenure_months,
                   {"monthly": 0, "annual": 1, "biennial": 2}[req.contract_type]]])
    prob = float(model.predict_proba(X)[0][1])
    pred = "churn" if prob > 0.5 else "retain"
    conf = "high" if abs(prob - 0.5) > 0.3 else "medium" if abs(prob - 0.5) > 0.1 else "low"

    response = {"churn_probability": round(prob, 4), "prediction": pred, "confidence": conf}
    background_tasks.add_task(log_prediction, req.dict(), response)

    logger.info(f"Inference time: {(time.perf_counter()-start)*1000:.2f}ms")
    return PredictionResponse(**response)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
async def metrics():
    # Expose for Prometheus scraping
    return {"requests_total": 1234, "avg_latency_ms": 23.4}
```

---

## Serialization Formats

```python
import joblib, pickle, torch, onnx

# ─── Sklearn / XGBoost / LightGBM ───────────────────────────────────────────
joblib.dump(model, "model.pkl")                 # Save
model = joblib.load("model.pkl")                # Load

# ─── PyTorch: state_dict (preferred) ─────────────────────────────────────────
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# ─── ONNX (cross-platform, fast inference) ───────────────────────────────────
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=17
)

import onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
output = sess.run(None, {"input": input_array})

# ─── Hugging Face ─────────────────────────────────────────────────────────────
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
model = AutoModel.from_pretrained("./my_model")
```

---

## Docker Commands

```bash
# Build
docker build -t my-api:1.0 .
docker build --no-cache -t my-api:1.0 .

# Run
docker run -d -p 8000:8000 --name api my-api:1.0
docker run -d -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/models:/app/models:ro \
  --memory="2g" --cpus="1.5" \
  my-api:1.0

# Debug
docker logs -f api
docker exec -it api bash
docker stats api

# Cleanup
docker system prune -af       # Remove all unused images + containers
```

---

## Cloud Platforms

### AWS SageMaker

```python
import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel

# Create model + endpoint
sm_model = SKLearnModel(
    model_data="s3://bucket/model.tar.gz",
    role=sagemaker.get_execution_role(),
    framework_version="1.2-1",
    entry_point="inference.py"
)
predictor = sm_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="churn-predictor-v2"
)

# Predict
result = predictor.predict([[35, 50000, 24, 1]])

# Auto-scaling
client = boto3.client("application-autoscaling")
client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/churn-predictor-v2/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1, MaxCapacity=10
)
```

### Hugging Face Spaces (Free GPU)

```python
# app.py (Gradio demo)
import gradio as gr
from transformers import pipeline

classifier = pipeline("text-classification", model="./model")

def predict(text):
    result = classifier(text)[0]
    return f"{result['label']} ({result['score']:.2%})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter text..."),
    outputs="text",
    title="Sentiment Classifier",
    examples=["I love this product!", "Terrible experience."]
)
demo.launch()
```

```
# requirements.txt + README.md with YAML frontmatter
---
title: My Classifier
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---
```

### GCP Vertex AI

```bash
# Build and push container
docker build -t gcr.io/PROJECT_ID/my-model:v1 .
docker push gcr.io/PROJECT_ID/my-model:v1

# Deploy to Vertex AI
gcloud ai endpoints create --display-name=my-endpoint --region=us-central1
gcloud ai models upload --display-name=my-model \
  --container-image-uri=gcr.io/PROJECT_ID/my-model:v1 \
  --container-predict-route=/predict \
  --container-health-route=/health
```

---

## A/B Testing

```python
import hashlib

def get_model_variant(user_id: str, traffic_split: float = 0.2) -> str:
    """Deterministic, consistent variant assignment per user."""
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    bucket = hash_val % 100
    return "model_b" if bucket < (traffic_split * 100) else "model_a"

# Statistical significance test
from scipy import stats

def check_significance(conversions_a, trials_a, conversions_b, trials_b, alpha=0.05):
    rate_a = conversions_a / trials_a
    rate_b = conversions_b / trials_b
    pooled = (conversions_a + conversions_b) / (trials_a + trials_b)
    se = (pooled * (1 - pooled) * (1/trials_a + 1/trials_b)) ** 0.5
    z_score = (rate_b - rate_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return {"p_value": p_value, "significant": p_value < alpha,
            "relative_lift": (rate_b - rate_a) / rate_a}

# Sample size calculator (before running experiment)
def required_samples(baseline_rate, min_detectable_effect, alpha=0.05, power=0.8):
    from statsmodels.stats.power import zt_ind_solve_power
    return zt_ind_solve_power(
        effect_size=(min_detectable_effect) / (baseline_rate * (1-baseline_rate))**0.5,
        alpha=alpha, power=power, alternative="two-sided"
    )
```

---

## Model Monitoring

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping
import pandas as pd

# Load reference (training) and current (production) data
reference = pd.read_parquet("reference_data.parquet")
current   = pd.read_parquet("production_last_7_days.parquet")

col_map = ColumnMapping(
    target="churn",
    prediction="prediction",
    numerical_features=["age", "income", "tenure_months"],
    categorical_features=["contract_type"]
)

report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
report.run(reference_data=reference, current_data=current, column_mapping=col_map)
report.save_html("monitoring.html")

# Programmatic check
result = report.as_dict()
if result["metrics"][0]["result"]["dataset_drift"]:
    # Trigger retraining pipeline
    import subprocess
    subprocess.run(["python", "src/train.py", "--retrain"])
```

---

## NGINX Config for ML API

```nginx
# /etc/nginx/sites-available/ml-api
upstream app {
    server 127.0.0.1:8000;
    keepalive 64;
}

limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/m;

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # Rate limiting
    limit_req zone=api_limit burst=10 nodelay;

    location / {
        proxy_pass         http://app;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 60s;
        client_max_body_size 10M;
    }

    location /health {
        proxy_pass http://app/health;
        access_log off;      # Don't log health checks
    }
}

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] Model serialized in production format (ONNX or joblib — not raw pickle if avoidable)
- [ ] Input validation with Pydantic models
- [ ] Unit tests: `pytest tests/test_api.py -v`
- [ ] Integration test: docker run + curl /predict
- [ ] Load test: `locust -f locustfile.py --headless -u 100 -r 10`
- [ ] Secrets in environment variables (not hardcoded)
- [ ] `.dockerignore` excludes data, notebooks, `.env`

### Post-Deployment
- [ ] Health check endpoint returns 200
- [ ] Latency monitored (p50, p95, p99)
- [ ] Error rate < 0.1%
- [ ] Data drift monitoring enabled
- [ ] Alerts configured (Grafana / PagerDuty)
- [ ] Rollback plan documented

### Production Readiness
- [ ] CI/CD pipeline runs tests on every PR
- [ ] Staging environment mirrors production
- [ ] Auto-scaling configured (min 1, max N instances)
- [ ] Logs shipping to centralized store (CloudWatch, Datadog)
- [ ] Model version tracked in responses (`model_version` field)
