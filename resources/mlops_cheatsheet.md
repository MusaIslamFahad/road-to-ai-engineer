# MLOps Cheatsheet

Quick reference for the full MLOps stack: experiment tracking, data versioning, CI/CD, orchestration, and monitoring.

---

## MLflow

### Experiment Tracking

```python
import mlflow
import mlflow.sklearn, mlflow.pytorch

mlflow.set_tracking_uri("http://localhost:5000")   # or "sqlite:///mlflow.db" for local
mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="xgboost-v3") as run:
    # Log hyperparameters
    mlflow.log_params({
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8
    })

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metrics({
        "train_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:,1]),
        "val_auc":   roc_auc_score(y_val,   model.predict_proba(X_val)[:,1]),
        "test_auc":  roc_auc_score(y_test,  model.predict_proba(X_test)[:,1]),
    })

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.html")

    # Log model + signature
    from mlflow.models import infer_signature
    sig = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "xgb-model", signature=sig,
                             registered_model_name="ChurnPredictor")

    print(f"Run ID: {run.info.run_id}")
```

### Autologging (Zero-Code Tracking)

```python
mlflow.autolog()    # Works for sklearn, xgboost, lightgbm, pytorch, keras

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
# ↑ Automatically logs params, metrics, model artifact
```

### Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promote to staging
client.transition_model_version_stage(
    name="ChurnPredictor", version=3, stage="Staging"
)

# Load latest production model
model_uri = "models:/ChurnPredictor/Production"
model = mlflow.sklearn.load_model(model_uri)

# Add description
client.update_model_version(
    name="ChurnPredictor", version=3,
    description="XGBoost with SMOTE; val AUC 0.923"
)
```

### Serving a Registered Model

```bash
# REST API endpoint
mlflow models serve -m "models:/ChurnPredictor/Production" --port 1234

# Predict
curl -X POST http://localhost:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"age": 35, "tenure": 12, "monthly_charges": 65.5}]}'
```

---

## DVC (Data Version Control)

### Setup

```bash
pip install dvc dvc-s3   # or dvc-gs, dvc-azure
git init
dvc init
dvc remote add -d myremote s3://my-ml-bucket/dvc-cache
```

### Track Data & Models

```bash
dvc add data/raw/train.csv          # Creates data/raw/train.csv.dvc
dvc add models/model.pkl
git add data/raw/train.csv.dvc .gitignore
git commit -m "Add training data"
dvc push                            # Upload to S3
```

### DVC Pipelines

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/train.csv
    outs:
      - data/prepared/train_clean.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/prepared/train_clean.csv
    params:
      - params.yaml:         # Tracks param changes
          - train.n_estimators
          - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json: {cache: false}

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/prepared/test_clean.csv
    metrics:
      - metrics/eval.json: {cache: false}
    plots:
      - reports/confusion_matrix.csv
```

```bash
dvc repro           # Run only changed stages
dvc dag             # Visualize pipeline DAG
dvc metrics show    # Compare metrics across commits
dvc params diff     # Show changed params
```

---

## GitHub Actions for ML

### CI Pipeline (`.github/workflows/ml-ci.yml`)

```yaml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov black isort

      - name: Lint
        run: |
          black --check src/
          isort --check-only src/

      - name: Run unit tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Validate data schema
        run: python tests/validate_data.py

  train-and-register:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Pull DVC data
        run: |
          pip install dvc dvc-s3
          dvc pull

      - name: Train model
        run: python src/train.py

      - name: Evaluate and gate
        run: |
          python src/evaluate.py
          python scripts/check_metrics.py --min-auc 0.85

      - name: Deploy to staging
        if: success()
        run: python scripts/deploy_staging.py
```

### Check Metrics Script

```python
# scripts/check_metrics.py
import json, argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument("--min-auc", type=float, default=0.80)
args = parser.parse_args()

with open("metrics/eval.json") as f:
    metrics = json.load(f)

auc = metrics["test_auc"]
print(f"Test AUC: {auc:.4f} (min required: {args.min_auc})")

if auc < args.min_auc:
    print("❌ Model quality gate FAILED")
    sys.exit(1)
print("✅ Model quality gate PASSED")
```

---

## Weights & Biases (W&B)

### Experiment Tracking

```python
import wandb

run = wandb.init(
    project="image-classification",
    name="resnet50-run-03",
    config={
        "architecture": "ResNet50",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 30,
        "optimizer": "AdamW"
    },
    tags=["resnet", "imagenet", "transfer-learning"]
)

for epoch in range(config.epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "lr": scheduler.get_last_lr()[0]
    })

    # Log images
    wandb.log({"predictions": [wandb.Image(img, caption=f"Pred: {pred}") for img, pred in samples]})

# Save model artifact
artifact = wandb.Artifact("resnet50-model", type="model")
artifact.add_file("model.pt")
run.log_artifact(artifact)
wandb.finish()
```

### Hyperparameter Sweeps

```python
# sweep_config.yaml
sweep_config = {
    "method": "bayes",   # "random", "grid", "bayes"
    "metric": {"name": "val/accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64, 128]},
        "dropout": {"distribution": "uniform", "min": 0.1, "max": 0.5},
        "hidden_size": {"values": [128, 256, 512]},
    }
}

def train_sweep():
    run = wandb.init()
    config = run.config
    model = build_model(config.hidden_size, config.dropout)
    train(model, lr=config.learning_rate, batch_size=config.batch_size)

sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train_sweep, count=50)
```

---

## Apache Kafka (Real-Time ML)

```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer — send events to topic
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

producer.send('transactions', {
    'user_id': 'u123',
    'amount': 150.0,
    'merchant': 'Amazon',
    'timestamp': '2025-01-15T10:30:00Z'
})
producer.flush()

# Consumer — ML model scores events in real-time
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    group_id='fraud-detector',
    auto_offset_reset='latest'
)

model = joblib.load('fraud_model.pkl')

for message in consumer:
    event = message.value
    features = extract_features(event)
    fraud_prob = model.predict_proba([features])[0][1]

    if fraud_prob > 0.8:
        alert_producer.send('fraud-alerts', {**event, 'fraud_score': fraud_prob})
```

---

## Model Monitoring with Evidently AI

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping

column_mapping = ColumnMapping(
    target="churn",
    prediction="prediction",
    numerical_features=["tenure", "monthly_charges", "total_charges"],
    categorical_features=["contract_type", "payment_method"]
)

# Generate monitoring report
report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])
report.run(
    reference_data=reference_df,     # Training data distribution
    current_data=production_df,       # Last 7 days of production data
    column_mapping=column_mapping
)
report.save_html("monitoring_report.html")

# Programmatic drift check
result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]
if drift_detected:
    send_slack_alert("⚠️ Data drift detected in churn model!")
```

---

## Quick Reference: Tool Selection

| Need | Tool | Why |
|---|---|---|
| Experiment tracking | MLflow or W&B | MLflow = self-hosted; W&B = team features |
| Data versioning | DVC | Git for data; integrates with S3/GCS |
| CI/CD | GitHub Actions | Free, integrates with everything |
| Pipeline orchestration | Airflow | Mature, widely used in industry |
| Feature store | Feast | Open-source, online + offline store |
| Model serving | MLflow / BentoML / TorchServe | MLflow = simple; BentoML = flexible |
| Monitoring | Evidently AI | Beautiful reports, open-source |
| Large-scale data | Apache Spark / PySpark | Distributed processing on clusters |
| Real-time streaming | Apache Kafka | Industry standard for event streaming |
