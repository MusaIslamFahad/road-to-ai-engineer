# Module 14: MLOps Basics

**Phase 8 — Production & Deployment** | Est. time: 1.5–2 months (full-time) · 3–4 months (part-time)

---

## Learning Objectives

By the end of this module you will:
- Version data and models with DVC alongside Git
- Track experiments, register models, and serve with MLflow
- Build CI/CD pipelines for ML with GitHub Actions
- Orchestrate training pipelines with Airflow
- Stream real-time data with Kafka and process at scale with Spark

---

## Prerequisites

- Module 13: Model Deployment

---

## 1. DVC — Data Version Control

```bash
pip install dvc dvc-s3
git init && dvc init

# Track data (not stored in Git — stored in S3/GCS)
dvc add data/train.csv                     # Creates data/train.csv.dvc
git add data/train.csv.dvc .gitignore
git commit -m "Track training data"

# Remote storage
dvc remote add -d myremote s3://my-bucket/dvc-cache
dvc push                                   # Upload to S3
dvc pull                                   # Download on another machine
```

### DVC Pipeline (dvc.yaml)
```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps: [src/prepare.py, data/raw/train.csv]
    outs: [data/processed/train_clean.csv]

  train:
    cmd: python src/train.py
    deps: [src/train.py, data/processed/train_clean.csv]
    params: [params.yaml]
    outs: [models/model.pkl]
    metrics: [metrics/scores.json: {cache: false}]

  evaluate:
    cmd: python src/evaluate.py
    deps: [src/evaluate.py, models/model.pkl, data/processed/test_clean.csv]
    metrics: [metrics/eval.json: {cache: false}]
```

```bash
dvc repro          # Run only changed stages
dvc dag            # Visualise pipeline DAG
dvc metrics show   # Compare metrics across commits
dvc metrics diff   # Diff between git commits
```

---

## 2. MLflow — Experiment Tracking & Registry

```python
import mlflow, mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction")
mlflow.autolog()   # Zero-code logging for sklearn/xgboost/lightgbm

# Manual tracking
with mlflow.start_run(run_name="xgb-balanced-v3") as run:
    mlflow.log_params({"n_estimators": 500, "max_depth": 6, "lr": 0.05})
    # ... train ...
    mlflow.log_metrics({"val_auc": 0.924, "test_auc": 0.918})
    mlflow.log_artifact("shap_plot.png", artifact_path="plots")
    mlflow.sklearn.log_model(model, "model",
                             registered_model_name="ChurnPredictor")

# Promote to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage("ChurnPredictor", version=3, stage="Production")

# Load production model
model = mlflow.sklearn.load_model("models:/ChurnPredictor/Production")
```

See full reference → [MLflow Comprehensive Guide](../resources/mlflow_comprehensive_guide.md)

---

## 3. GitHub Actions CI/CD for ML

```yaml
# .github/workflows/ml-pipeline.yml
name: ML CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: "3.10", cache: "pip"}
      - run: pip install -r requirements.txt pytest black
      - run: black --check src/
      - run: pytest tests/ -v --tb=short

  train-and-register:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
      AWS_ACCESS_KEY_ID:     ${{ secrets.AWS_KEY }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: "3.10"}
      - run: pip install -r requirements.txt dvc dvc-s3
      - run: dvc pull                         # Pull latest data
      - run: python src/train.py              # Train + log to MLflow
      - name: Quality gate
        run: |
          python scripts/check_metrics.py --min-auc 0.88 || exit 1
      - run: python scripts/deploy_staging.py
```

### Quality Gate Script
```python
# scripts/check_metrics.py
import json, argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument("--min-auc", type=float, default=0.85)
args = parser.parse_args()

metrics = json.load(open("metrics/eval.json"))
auc = metrics["test_auc"]
print(f"Test AUC: {auc:.4f}  (min: {args.min_auc})")
if auc < args.min_auc:
    print("❌ Quality gate FAILED"); sys.exit(1)
print("✅ Quality gate PASSED")
```

---

## 4. Weights & Biases (W&B)

```python
import wandb

run = wandb.init(project="churn-prediction",
                 config={"lr": 1e-3, "batch_size": 64, "epochs": 30})
for epoch in range(30):
    train_loss, val_loss = train_epoch(), evaluate()
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

# Hyperparameter sweep
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_auc", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="churn-prediction")
wandb.agent(sweep_id, function=train_fn, count=50)
```

---

## 5. Apache Kafka — Real-Time ML

```python
from kafka import KafkaProducer, KafkaConsumer
import json, joblib

# Producer — send events
producer = KafkaProducer(bootstrap_servers=["localhost:9092"],
                         value_serializer=lambda v: json.dumps(v).encode())
producer.send("transactions", {"user_id": "u123", "amount": 150.0,
                                "merchant": "Amazon"})

# Consumer — ML scoring in real-time
model    = joblib.load("fraud_model.pkl")
consumer = KafkaConsumer("transactions",
                         bootstrap_servers=["localhost:9092"],
                         value_deserializer=lambda v: json.loads(v.decode()),
                         group_id="fraud-scorer")

for msg in consumer:
    event    = msg.value
    features = extract_features(event)
    score    = model.predict_proba([features])[0][1]
    if score > 0.8:
        producer.send("fraud-alerts", {**event, "fraud_score": score})
```

---

## 6. Apache Spark / PySpark — Large-Scale ML

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("ChurnML").getOrCreate()

df = spark.read.csv("s3://bucket/churn_data.csv", header=True, inferSchema=True)

# Spark ML Pipeline
assembler = VectorAssembler(inputCols=["age","tenure","charges"], outputCol="features")
scaler    = StandardScaler(inputCol="features", outputCol="scaled_features")
lr        = LogisticRegression(featuresCol="scaled_features", labelCol="churn")
pipeline  = Pipeline(stages=[assembler, scaler, lr])

model = pipeline.fit(df)
predictions = model.transform(df)
predictions.select("churn", "prediction", "probability").show(5)

# MLlib at scale
from pyspark.mllib.evaluation import BinaryClassificationMetrics
metrics = BinaryClassificationMetrics(
    predictions.select("probability", "churn").rdd.map(
        lambda r: (float(r["probability"][1]), float(r["churn"]))
    )
)
print(f"AUC: {metrics.areaUnderROC:.4f}")
spark.stop()
```

---

## 7. Airflow — Pipeline Orchestration

```python
# dags/churn_retrain_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {"owner": "ml-team", "retries": 2,
                "retry_delay": timedelta(minutes=5)}

with DAG("churn_retrain", default_args=default_args,
         schedule_interval="@weekly", start_date=datetime(2025, 1, 1),
         catchup=False, tags=["ml", "churn"]) as dag:

    extract = PythonOperator(task_id="extract_data",
                             python_callable=extract_from_db)
    transform = PythonOperator(task_id="feature_engineering",
                               python_callable=build_features)
    train    = PythonOperator(task_id="train_model",
                              python_callable=train_and_log)
    evaluate = PythonOperator(task_id="evaluate_and_gate",
                              python_callable=quality_gate)
    deploy   = PythonOperator(task_id="deploy_to_staging",
                              python_callable=deploy_staging)

    extract >> transform >> train >> evaluate >> deploy
```

---

## Related Resources

- [MLOps Cheatsheet](../resources/mlops_cheatsheet.md)
- [MLflow Comprehensive Guide](../resources/mlflow_comprehensive_guide.md)
- [Docker Tutorial](../resources/docker_tutorial.md)
- [Git Guide](../resources/git_guide.md)

**[← Module 13](../13-model-deployment/README.md)** | **[→ Module 15](../15-time-series-analysis/README.md)**
