# Advanced Projects

**9 Projects** | Prerequisites: Phases 0–8 | Est. time per project: 1–3 weeks

These are **portfolio-grade** projects. Prioritise quality over quantity — one excellent project beats five rushed ones.

---

## Project 1: Image Classification — CIFAR-10

**Skills**: CNNs, ResNet, EfficientNet, transfer learning, data augmentation, mixed precision  
**Dataset**: `torchvision.datasets.CIFAR10` — 60,000 images, 10 classes  
**Target metric**: Test accuracy > 93% (human-level ≈ 94%)

**Steps**:
1. Baseline: simple 3-layer CNN from scratch → note accuracy (~75%)
2. Data augmentation: `RandomHorizontalFlip`, `RandomCrop(32, padding=4)`, `ColorJitter`, `CutMix`
3. Transfer learning: fine-tune `EfficientNet-B0` pre-trained on ImageNet
4. Training tricks: label smoothing (0.1), `OneCycleLR`, mixed precision (`torch.cuda.amp`)
5. Test-time augmentation (TTA): average predictions over 5 augmented views
6. Visualise: Grad-CAM heatmaps for correct and incorrect predictions

**Stretch**: Train a ViT (Vision Transformer) from scratch; compare to EfficientNet

---

## Project 2: Sentiment Analysis on Product Reviews

**Skills**: Fine-tuning BERT, HuggingFace Trainer, LoRA, model deployment  
**Dataset**: [Amazon Product Reviews](https://huggingface.co/datasets/amazon_us_reviews) or [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full) — 5-class sentiment  
**Target metric**: F1-macro > 0.68

**Steps**:
1. Baseline: TF-IDF + Logistic Regression (fast, interpretable)
2. Fine-tune `distilbert-base-uncased` with HuggingFace Trainer
3. Fine-tune `roberta-base` — compare with DistilBERT
4. Apply LoRA (r=16) to reduce trainable params by 95%; compare quality
5. Evaluate: classification report + confusion matrix per star rating
6. Build FastAPI `/predict` endpoint; wrap in Docker; deploy to Railway

---

## Project 3: Advanced Time Series Forecasting

**Skills**: LightGBM with features, NHITS/N-BEATS, ensemble, hierarchical forecasting  
**Dataset**: [M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) — Walmart sales  
**Target metric**: WRMSSE (competition metric)

**Steps**:
1. EDA — 30,490 item-store combinations, calendar events, SNAP flags, prices
2. Feature engineering: extensive lag + rolling features per item-store level
3. LightGBM with recursive multi-step forecasting
4. NeuralForecast `NHITS` model for direct multi-step
5. Hierarchical reconciliation: bottom-up vs. top-down
6. Ensemble LightGBM + NHITS; evaluate per-level (item / store / national)

---

## Project 4: LLM Chatbot & RAG System

**Skills**: RAG, LangChain, ChromaDB/Pinecone, RAGAS evaluation, FastAPI streaming  
**Dataset**: Your own documents (PDFs, web pages, or a public corpus)  
**Target metric**: RAGAS faithfulness > 0.85, context recall > 0.80

**Steps**:
1. Ingest documents: PDF + web pages → `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap)
2. Embed with `text-embedding-3-small`; store in ChromaDB
3. Implement hybrid retrieval: dense + BM25 via `EnsembleRetriever`
4. Add cross-encoder re-ranking (`BAAI/bge-reranker-large`)
5. Build conversational RAG with chat history
6. Streaming FastAPI endpoint with SSE
7. Evaluate with RAGAS on 50-question golden dataset
8. Add LangSmith tracing for observability

**Stretch**: Add an AI agent layer — tool use, web search, Python execution

---

## Project 5: Object Detection — Custom Dataset

**Skills**: YOLO v8, data labelling, mAP evaluation, ONNX export, deployment  
**Dataset**: Your own photos **or** [COCO subset](https://cocodataset.org/) or [Open Images](https://storage.googleapis.com/openimages/web/index.html)  
**Target metric**: mAP@50 > 0.70 on your validation set

**Steps**:
1. Collect 200–500 images for your chosen objects (or use existing dataset)
2. Label with [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com/)
3. Train YOLOv8n (nano) → YOLOv8s (small) → compare mAP vs. inference speed
4. Data augmentation in YOLO config: mosaic, flipping, colour jitter
5. Export to ONNX; measure inference latency on CPU
6. Build a Gradio demo: upload image → see bounding boxes; deploy to HF Spaces

---

## Project 6: End-to-End ML Pipeline with Full MLOps

**Skills**: DVC, MLflow, GitHub Actions, Docker, AWS/Railway, Evidently AI monitoring  
**Dataset**: Any dataset from previous projects  
**Target**: A fully automated, observable, continuously-deployed ML system

**Pipeline**:
```
GitHub commit → CI tests → DVC data pull → Train → MLflow log
→ Quality gate → Docker build → Push to ECR → Deploy to ECS
→ Evidently monitoring → Drift alert → Auto-retrain trigger
```

**Steps**:
1. Version data with DVC; store in S3
2. Track experiments with MLflow (autolog + manual metrics)
3. GitHub Actions: lint → test → train → quality gate → deploy on merge to main
4. Dockerise FastAPI serving app; push to Docker Hub / ECR
5. Deploy to Railway or AWS ECS
6. Set up Evidently AI weekly monitoring report
7. Alerting: email/Slack when drift score exceeds threshold

---

## Project 7: Generative Model (GAN or VAE)

**Skills**: DCGAN, VAE, FID score, latent space interpolation, conditional generation  
**Dataset**: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (faces) or [MNIST Fashion](https://huggingface.co/datasets/zalando-datasets/fashion_mnist)  
**Target metric**: FID score < 50 (lower = better image quality)

### Option A — DCGAN
```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim → 512×4×4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output: (3, 32, 32), values ∈ [-1,1]
        )
```

### Option B — VAE
```python
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ...   # → mu and log_var
        self.decoder = ...   # latent → image

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        return self.decoder(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    recon = F.mse_loss(recon_x, x, reduction="sum")
    kld   = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kld
```

**Stretch**: Conditional GAN — control which digit/face attribute to generate

---

## Project 8: Model Explainability Dashboard

**Skills**: SHAP, LIME, PDP, ICE, Evidently AI, interactive Streamlit dashboard  
**Dataset**: Any tabular classification model from previous projects  
**Target**: A Streamlit app that non-technical stakeholders can use

**Dashboard features**:
- Global: SHAP beeswarm + feature importance bar chart
- Local: paste new input → waterfall SHAP + LIME text explanation + decision path
- Drift: upload new data → Evidently data drift report embedded in app
- Calibration: reliability diagram + Brier score

---

## Project 9: Production Model Serving at Scale

**Skills**: FastAPI, ONNX Runtime, Redis caching, NGINX, load testing, Prometheus/Grafana  
**Dataset**: Any trained model  
**Target**: < 30ms p99 latency under 100 req/s

**Steps**:
1. Export model to ONNX; measure raw inference latency
2. Build FastAPI with async endpoints + Pydantic validation
3. Add Redis semantic caching (identical inputs → return cached result)
4. Rate limiting with `slowapi`; API key authentication with `fastapi-users`
5. Containerise; deploy 3 replicas behind NGINX
6. Load test with Locust: ramp to 100 users; monitor p50/p95/p99
7. Instrument with Prometheus metrics; visualise in Grafana
8. Write a 1-page architecture decision record (ADR) documenting trade-offs

---

## Capstone Projects (portfolio centrepiece — pick one)

### ML Engineer Capstone
Full churn prediction pipeline: data → features → XGBoost + LightGBM ensemble → FastAPI → Docker → AWS ECS → MLflow tracking → GitHub Actions CI/CD → Evidently monitoring → auto-retrain. Write a 2-page technical blog post documenting it.

### LLM/RAG Capstone
Multi-document knowledge base chatbot: hybrid RAG (dense + BM25 + reranking) → RAGAS evaluation pipeline → streaming FastAPI → LangSmith observability → Gradio demo on HF Spaces. Blog post with benchmark results comparing naive RAG vs. advanced RAG.

### Computer Vision Capstone
End-to-end object detection: data collection + labelling → YOLOv8 training → ONNX export → FastAPI inference server → Docker → deployed Gradio demo. Measure and report mAP@50, FPS on CPU, FPS on GPU.

**[← Intermediate Projects](../17-projects-intermediate/README.md)** | **[← Module 22](../22-reinforcement-learning/README.md)**
