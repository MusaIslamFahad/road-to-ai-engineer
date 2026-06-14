# MLflow Comprehensive Guide

End-to-end reference for MLflow: experiment tracking, model registry, serving, and multi-model projects.

---

## Setup & Tracking Server

```bash
pip install mlflow

# Local filesystem (simplest — great for solo projects)
mlflow ui                          # Opens at http://localhost:5000

# PostgreSQL backend + S3 artifacts (team / production)
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost:5432/mlflowdb \
  --default-artifact-root s3://my-ml-bucket/mlflow-artifacts \
  --host 0.0.0.0 --port 5000

# SQLite (single-machine team)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```

```python
import mlflow

# Set tracking server
mlflow.set_tracking_uri("http://localhost:5000")
# or: mlflow.set_tracking_uri("sqlite:///mlflow.db")
# or: set env var MLFLOW_TRACKING_URI

mlflow.set_experiment("churn-prediction-v2")
```

---

## Part 1: Experiment Tracking

### Manual Logging

```python
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

with mlflow.start_run(run_name="rf-balanced-v3",
                      tags={"team": "ml-platform", "env": "dev"}) as run:

    # ─── Log parameters ──────────────────────────────────────────────────────
    params = {
        "n_estimators": 300, "max_depth": 10,
        "min_samples_leaf": 5, "class_weight": "balanced"
    }
    mlflow.log_params(params)

    # ─── Train ───────────────────────────────────────────────────────────────
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # ─── Log metrics ─────────────────────────────────────────────────────────
    for split, X_s, y_s in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = model.predict_proba(X_s)[:, 1]
        pred  = model.predict(X_s)
        mlflow.log_metrics({
            f"{split}_auc":      roc_auc_score(y_s, proba),
            f"{split}_pr_auc":   average_precision_score(y_s, proba),
            f"{split}_f1_macro": f1_score(y_s, pred, average="macro"),
        })

    # ─── Log artifacts ───────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import shap, json

    # SHAP feature importance plot
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    shap.summary_plot(shap_vals, X_test, show=False)
    plt.savefig("shap_summary.png", bbox_inches="tight")
    mlflow.log_artifact("shap_summary.png", artifact_path="plots")

    # Training config
    with open("config.json", "w") as f:
        json.dump({"feature_cols": list(X_train.columns), "target": "churn"}, f)
    mlflow.log_artifact("config.json")

    # ─── Log model with signature and input example ──────────────────────────
    from mlflow.models import infer_signature
    signature    = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:5]

    mlflow.sklearn.log_model(
        model, "random-forest-model",
        signature=signature,
        input_example=input_example,
        registered_model_name="ChurnPredictor"   # Automatically registers
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")
```

### Autologging (Zero Extra Code)

```python
import mlflow

# Enable autologging for sklearn/xgboost/lightgbm/keras/pytorch
mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=False
)

# Now just train normally — everything is logged automatically
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)
# ↑ MLflow automatically logs: params, metrics, feature importance, model artifact

# XGBoost autolog
import xgboost as xgb
mlflow.xgboost.autolog()
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
# ↑ Also logs validation curves at each boosting round
```

### Logging Epoch-by-Epoch Metrics (Deep Learning)

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"epochs": 50, "lr": 1e-3, "batch_size": 64})

    for epoch in range(50):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc     = evaluate(model, val_loader)

        # Log at each step
        mlflow.log_metrics({
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc
        }, step=epoch)

    # Log final model
    mlflow.pytorch.log_model(model, "pytorch-model")
```

---

## Part 2: Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# ─── Register a model ──────────────────────────────────────────────────────
mlflow.sklearn.log_model(model, "model", registered_model_name="ChurnPredictor")

# Or register from an existing run
run_id = "abc123"
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "ChurnPredictor")
print(f"Version: {mv.version}")

# ─── Add description and tags ─────────────────────────────────────────────
client.update_registered_model(
    name="ChurnPredictor",
    description="XGBoost churn predictor trained on Q3 2025 data. AUC 0.924."
)
client.set_registered_model_tag("ChurnPredictor", "owner", "ml-platform-team")
client.update_model_version(
    name="ChurnPredictor", version=3,
    description="Best version: val AUC 0.924, PR-AUC 0.711"
)

# ─── Promote through stages ────────────────────────────────────────────────
# Stages: None → Staging → Production → Archived
client.transition_model_version_stage("ChurnPredictor", version=3, stage="Staging")
client.transition_model_version_stage("ChurnPredictor", version=3, stage="Production")
client.transition_model_version_stage("ChurnPredictor", version=2, stage="Archived")

# ─── Load production model ─────────────────────────────────────────────────
prod_model_uri = "models:/ChurnPredictor/Production"
loaded_model   = mlflow.sklearn.load_model(prod_model_uri)
preds = loaded_model.predict(X_test)

# Load specific version
v3_model = mlflow.sklearn.load_model("models:/ChurnPredictor/3")
```

---

## Part 3: Querying Experiments Programmatically

```python
import mlflow
import pandas as pd

# Search runs in an experiment
runs = mlflow.search_runs(
    experiment_names=["churn-prediction-v2"],
    filter_string="metrics.val_auc > 0.90 AND params.n_estimators = '300'",
    order_by=["metrics.val_auc DESC"],
    max_results=10
)
print(runs[["run_id", "metrics.val_auc", "metrics.val_pr_auc", "params.max_depth"]])

# Get best run
best_run = runs.iloc[0]
print(f"Best run: {best_run['run_id']} | Val AUC: {best_run['metrics.val_auc']:.4f}")

# Compare two runs
run_ids = [runs.iloc[0]["run_id"], runs.iloc[1]["run_id"]]
comparison = mlflow.search_runs(filter_string=f"run_id IN {tuple(run_ids)}")
```

---

## Part 4: MLflow Projects

```yaml
# MLproject file
name: churn-prediction

conda_env: conda.yaml   # or python_env.yaml

entry_points:
  main:
    parameters:
      n_estimators:  {type: int,   default: 300}
      max_depth:     {type: int,   default: 10}
      learning_rate: {type: float, default: 0.05}
      data_path:     {type: str,   default: "data/train.csv"}
    command: "python train.py --n_estimators {n_estimators} --max_depth {max_depth} --learning_rate {learning_rate} --data_path {data_path}"

  evaluate:
    parameters:
      model_uri: {type: str}
      test_path: {type: str, default: "data/test.csv"}
    command: "python evaluate.py --model_uri {model_uri} --test_path {test_path}"
```

```bash
# Run project locally
mlflow run . -P n_estimators=500 -P max_depth=8

# Run from GitHub
mlflow run https://github.com/user/churn-project -P n_estimators=500

# Run on Databricks / SageMaker
mlflow run . --backend databricks --backend-config cluster.json
```

---

## Part 5: Model Serving

```bash
# Serve a registered model as REST API
mlflow models serve \
  -m "models:/ChurnPredictor/Production" \
  --port 1234 \
  --workers 4

# Test the endpoint
curl -X POST http://localhost:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"age": 35, "tenure_months": 12, "monthly_charges": 65.5}]}'
# Response: {"predictions": [0.23]}

# Build Docker image from registered model
mlflow models build-docker \
  -m "models:/ChurnPredictor/Production" \
  -n "churn-predictor:latest" \
  --enable-mlserver

# Run the Docker image
docker run -p 8080:8080 churn-predictor:latest
```

---

## Part 6: Best Practices

```python
# ─── Use consistent run naming ─────────────────────────────────────────────
import datetime
run_name = f"xgb-v{datetime.date.today().strftime('%Y%m%d')}-lr{lr}-depth{depth}"

# ─── Log git commit for reproducibility ────────────────────────────────────
import subprocess
git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
mlflow.set_tag("git_commit", git_hash)

# ─── Nested runs for hyperparameter search ─────────────────────────────────
with mlflow.start_run(run_name="hparam-search") as parent_run:
    for trial in study.trials:
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("val_auc", trial.value)

# ─── Context manager for safe cleanup ──────────────────────────────────────
import atexit
mlflow.end_run()   # Ensure run is closed even if exception occurs
```

| Tip | Why |
|---|---|
| Use `mlflow.autolog()` first | Zero-effort logging; add manual for extras |
| Always log `input_example` | Enables schema validation on serving |
| Set `registered_model_name` | Enables Model Registry tracking |
| Log config files as artifacts | Full reproducibility |
| Tag runs with `git_commit` | Know exactly what code produced each run |
| Use nested runs for sweeps | Clean hierarchy in the UI |
