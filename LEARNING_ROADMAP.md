# AI Engineer Learning Roadmap

This document provides a detailed, week-by-week roadmap for becoming a Generalist AI Engineer. Use it alongside the main README as your daily/weekly planner.

---

## How to Use This Roadmap

- **Full-Time (30–40 hrs/week)**: Follow the full-time estimates
- **Part-Time (10–15 hrs/week)**: Multiply full-time estimates by ~2.5×
- **Background credit**: Prior programming experience saves 2–4 months; prior math background saves 1–2 months
- **Don't skip projects**: They are not optional for AI Engineers — portfolio is everything

---

## Phase 0: Foundation (~2–3 months full-time)

### Python Programming
- [ ] Variables, data types, control flow, functions
- [ ] OOP: classes, inheritance, polymorphism, encapsulation
- [ ] Iterators, generators, decorators, context managers
- [ ] File I/O, JSON, pickle, exception handling
- [ ] Time complexity, Big O notation
- [ ] **Project**: Movie script generator

### Mathematics
- [ ] Linear Algebra: vectors, matrices, matrix multiplication, eigenvalues, SVD
- [ ] Statistics: mean/variance/std, distributions, hypothesis testing, Bayes' theorem
- [ ] Calculus: derivatives, partial derivatives, chain rule, gradient descent intuition
- [ ] Probability theory: conditional probability, independence, Bayes' theorem

### Environment Setup
- [ ] Python environment (Anaconda or venv)
- [ ] Jupyter Notebook
- [ ] Git & GitHub basics
- [ ] VS Code or PyCharm setup

**Checkpoint**: Can you write a Python class with inheritance, handle exceptions, and explain gradient descent mathematically?

---

## Phase 1: Data Fundamentals (~2–3 months full-time)

### NumPy
- [ ] ndarray creation, dtypes, shape, reshape
- [ ] Indexing, slicing, fancy indexing, boolean masking
- [ ] Broadcasting rules and vectorized operations
- [ ] Linear algebra operations: dot, matmul, solve, eig, svd
- [ ] Random module and statistical functions

### Pandas
- [ ] Series and DataFrame creation
- [ ] Data loading: CSV, Excel, JSON, SQL
- [ ] Indexing: loc, iloc, boolean, at, iat
- [ ] Data cleaning: missing values, duplicates, dtypes
- [ ] GroupBy, aggregation, apply, transform
- [ ] Merge, join, concat, pivot tables
- [ ] Time-indexed DataFrames

### Visualization
- [ ] Matplotlib: figure, axes, subplots, customization
- [ ] Seaborn: distribution plots, categorical plots, heatmaps
- [ ] Plotly: interactive charts and dashboards
- [ ] Streamlit: ML application dashboards

### EDA
- [ ] Univariate and bivariate analysis
- [ ] Correlation analysis
- [ ] Outlier detection
- [ ] Data profiling with pandas-profiling / ydata-profiling

**Checkpoint**: Can you load a messy CSV, clean it, explore it with EDA, and build a Streamlit dashboard?

---

## Phase 2: ML Basics (~3–4 months full-time)

### Module 02 – Introduction to ML
- [ ] Types of ML: supervised, unsupervised, reinforcement
- [ ] ML workflow: problem definition → data → features → model → evaluate → deploy
- [ ] Overfitting, underfitting, bias-variance tradeoff
- [ ] **Project**: First end-to-end ML model

### Module 03 – Regression
- [ ] Simple and multiple linear regression
- [ ] Polynomial regression
- [ ] Ridge, Lasso, ElasticNet regularization
- [ ] Gradient descent variants (batch, SGD, mini-batch)
- [ ] Statistical regression analysis (statsmodels)

### Module 04 – Classification
- [ ] Logistic regression (binary, multiclass)
- [ ] K-Nearest Neighbours
- [ ] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [ ] Decision Trees (entropy, Gini, pruning)
- [ ] Support Vector Machines (linear, RBF, polynomial kernels)
- [ ] Confusion matrix, precision, recall, F1, ROC-AUC

### Module 05 – Model Evaluation & Optimization
- [ ] Train/val/test split strategy
- [ ] K-Fold, Stratified K-Fold, Leave-One-Out, Time Series CV
- [ ] Data leakage: types and prevention
- [ ] Hyperparameter tuning: GridSearch, RandomSearch, Optuna
- [ ] Learning curves and validation curves
- [ ] Model calibration: Platt scaling, isotonic regression

**Checkpoint**: Can you build an end-to-end ML pipeline with proper validation, handle hyperparameter tuning, and avoid data leakage?

---

## Phase 3: Advanced Supervised Learning (~2–3 months full-time)

### Module 06 – Ensemble Methods
- [ ] Bagging: random subsampling, Random Forest
- [ ] Boosting: AdaBoost → Gradient Boosting → XGBoost → LightGBM → CatBoost
- [ ] Stacking and meta-learners
- [ ] Voting ensembles (hard and soft)
- [ ] Feature importance from tree models

### Module 07 – Feature Engineering
- [ ] Feature selection: filter, wrapper, embedded methods
- [ ] Encoding: One-Hot, Label, Target, Frequency, Binary, WOE
- [ ] Scaling: StandardScaler, MinMaxScaler, RobustScaler
- [ ] Transformation: log, sqrt, Box-Cox, Yeo-Johnson
- [ ] PCA for dimensionality reduction
- [ ] sklearn Pipeline and ColumnTransformer patterns
- [ ] Decision Tree-based binning, custom transformers

**Checkpoint**: Can you build a production-ready sklearn Pipeline with custom transformers and compete on a Kaggle tabular dataset?

---

## Phase 4: Unsupervised Learning (~1–2 months full-time)

### Module 08 – Unsupervised Learning
- [ ] K-Means: algorithm, elbow method, K-Means++
- [ ] Hierarchical clustering: Ward, complete, average linkage; dendrograms
- [ ] DBSCAN and HDBSCAN: density-based clustering
- [ ] Evaluation: silhouette score, Davies-Bouldin, Calinski-Harabasz
- [ ] PCA: variance explained, scree plots, biplots
- [ ] t-SNE: perplexity, early exaggeration, visualization
- [ ] UMAP: faster, better for downstream tasks
- [ ] SVD and matrix factorization
- [ ] Anomaly detection: Isolation Forest, One-Class SVM, LOF
- [ ] Association rules: Apriori, FP-Growth, support/confidence/lift

**Checkpoint**: Can you cluster a customer dataset, reduce dimensions with PCA/UMAP, and detect anomalies?

---

## Phase 5: Deep Learning Fundamentals (~2–3 months full-time)

### Module 09 – Neural Networks Basics
- [ ] Perceptron → MLP: layer structure, forward pass
- [ ] Activation functions: sigmoid, tanh, ReLU, LeakyReLU, GELU, Swish
- [ ] Loss functions: MSE, cross-entropy, binary cross-entropy, focal loss
- [ ] Backpropagation and chain rule: manual gradient computation
- [ ] Optimizers: SGD, Momentum, Adam, AdaGrad, RMSprop, AdamW
- [ ] Regularization: Dropout, L1/L2, Early Stopping
- [ ] Batch Normalization and Layer Normalization
- [ ] Weight initialization: Xavier, He/Kaiming
- [ ] Vanishing/exploding gradients; gradient clipping

### Module 10 – Deep Learning Frameworks
- [ ] PyTorch: tensors, autograd, nn.Module, DataLoader, training loop
- [ ] Custom datasets and data augmentation pipelines
- [ ] TensorFlow/Keras: Sequential, Functional API, custom layers
- [ ] Mixed precision training (float16/bfloat16)
- [ ] Model saving: state_dict, .h5, ONNX, TorchScript
- [ ] GPU training; moving tensors between devices
- [ ] Weights & Biases: experiment tracking, hyperparameter sweeps

**Checkpoint**: Can you implement a custom neural network in PyTorch, train it with a proper loop, track experiments in W&B, and export it to ONNX?

---

## Phase 6: Specialized Deep Learning (~5–7 months full-time)

### Module 11 – Computer Vision (~2.5–3.5 months)
- [ ] Image basics: channels, normalization, augmentation (albumentations, torchvision)
- [ ] Convolutions: filters, padding, stride, receptive field
- [ ] CNN architectures: LeNet → AlexNet → VGG → ResNet → EfficientNet → ConvNeXt
- [ ] Transfer learning: feature extraction and fine-tuning strategies
- [ ] Object Detection:
  - [ ] Two-stage: R-CNN → Fast R-CNN → Faster R-CNN
  - [ ] One-stage: YOLO (v3 → v8), SSD, RetinaNet (focal loss)
  - [ ] Evaluation: mAP, IoU, NMS
- [ ] Semantic Segmentation: FCN, U-Net, DeepLab
- [ ] Instance Segmentation: Mask R-CNN
- [ ] Generative Models:
  - [ ] GANs: DCGAN, conditional GAN, CycleGAN, StyleGAN
  - [ ] VAEs: encoder-decoder, reparameterization trick
  - [ ] Diffusion Models: DDPM, score matching, Stable Diffusion
- [ ] Vision Transformers (ViT, DEIT, Swin Transformer)

### Module 12 – Natural Language Processing (~2.5–3.5 months)
- [ ] Text preprocessing: tokenization, stemming, lemmatization, stop words
- [ ] Classical NLP: Bag-of-Words, TF-IDF, n-grams
- [ ] Word embeddings: Word2Vec (CBOW, Skip-gram), GloVe, FastText
- [ ] Sequence models: RNN → LSTM/GRU → bidirectional
- [ ] Sequence-to-sequence; attention mechanism (Bahdanau, Luong)
- [ ] Transformer architecture: self-attention, multi-head attention, positional encoding
- [ ] BERT: pre-training (MLM, NSP), fine-tuning, sentence embeddings
- [ ] GPT architecture: autoregressive language modeling, causal attention
- [ ] T5, RoBERTa, DeBERTa, modern LLM architectures
- [ ] Hugging Face: Transformers, Datasets, Evaluate, Trainer API
- [ ] Fine-tuning strategies: full fine-tuning, LoRA, QLoRA, prefix tuning
- [ ] RAG: document chunking, embedding, retrieval, re-ranking

**Checkpoint**: Can you build an object detector with YOLO, fine-tune BERT for classification, and implement a simple RAG pipeline?

---

## Branch: Time Series (~0.5–1 month full-time)

*Take this after Phase 5 or during Phase 6, in parallel*

### Module 15 – Time Series Analysis
- [ ] Trend, seasonality, noise decomposition (additive vs. multiplicative)
- [ ] Stationarity: ADF test, KPSS test, differencing
- [ ] ACF and PACF interpretation
- [ ] Classical models: ARIMA, SARIMA, SARIMAX, Exponential Smoothing (Holt-Winters)
- [ ] ML for time series: lag features, rolling statistics, date/time features
- [ ] Deep sequence models: LSTM, GRU for forecasting
- [ ] Evaluation: MAE, RMSE, MAPE; walk-forward validation (no data leakage)
- [ ] Prophet, NeuralForecast, Darts frameworks

**Checkpoint**: Can you decompose a time series, fit ARIMA, compare against LSTM, and evaluate with walk-forward validation?

---

## Phase 7: Generative AI & Modern LLMs (~1–2 months full-time)

### Module 25 – Generative AI & LLMs
- [ ] Prompt engineering: zero-shot, few-shot, chain-of-thought, ReAct, structured output
- [ ] LLM API usage: OpenAI, Anthropic, Cohere, Google Gemini, local models (Ollama)
- [ ] Embeddings: semantic similarity, cosine distance, batch encoding
- [ ] Vector databases: Pinecone, ChromaDB, FAISS, Weaviate, Qdrant
- [ ] RAG systems:
  - [ ] Document loading, chunking strategies (fixed, recursive, semantic)
  - [ ] Hybrid retrieval (dense + sparse BM25)
  - [ ] Re-ranking (cross-encoders, Cohere Rerank)
  - [ ] Evaluation: RAGAS, faithfulness, relevancy, context recall
- [ ] LangChain: chains, LCEL, memory, document loaders, vector stores
- [ ] LlamaIndex: document indexing, query engines, agents
- [ ] AI Agents: tool use, ReAct pattern, planning, reflection loops
- [ ] Multi-agent systems: CrewAI (roles, tasks, crews), AutoGen (group chat), LangGraph (state machines)
- [ ] Model Context Protocol (MCP): tool servers, client integration
- [ ] LLM fine-tuning: LoRA, QLoRA, PEFT, DPO
- [ ] Evaluation and monitoring: LangSmith, Langfuse, hallucination detection

**Checkpoint**: Can you build a production RAG system with hybrid retrieval, implement a multi-agent workflow, and evaluate LLM outputs?

---

## Phase 7.5: Essential Skills (~2–3 months full-time, can overlap)

### Module 19 – SQL & Databases
- [ ] SQL basics: SELECT, WHERE, GROUP BY, HAVING, ORDER BY
- [ ] Joins: INNER, LEFT, RIGHT, FULL OUTER, self-join, cross-join
- [ ] Subqueries, CTEs (WITH clause), recursive CTEs
- [ ] Window functions: ROW_NUMBER, RANK, LEAD, LAG, SUM OVER
- [ ] Stored procedures, views, indexes, query optimization (EXPLAIN)
- [ ] Data cleaning with SQL
- [ ] Python + SQL: SQLAlchemy, psycopg2, sqlite3
- [ ] NoSQL: MongoDB (aggregation pipeline), Redis (caching), Cassandra (wide-column), Neo4j (graph)
- [ ] Vector databases for AI: pgvector, Pinecone, Weaviate

### Module 20 – Handling Imbalanced Data
- [ ] Understanding class imbalance (1:10, 1:100, 1:1000 ratios)
- [ ] Oversampling: SMOTE, ADASYN, Borderline-SMOTE
- [ ] Undersampling: Random, Tomek Links, Edited Nearest Neighbours
- [ ] Combined: SMOTEENN, SMOTETomek
- [ ] Class weights and cost-sensitive learning
- [ ] Threshold tuning for optimal F1 / business metric
- [ ] Evaluation: PR-AUC, F1 macro, G-mean, MCC

### Module 21 – Model Explainability
- [ ] Global vs. local interpretability
- [ ] Feature importance: impurity-based (tree), permutation importance
- [ ] SHAP: TreeSHAP, KernelSHAP, DeepSHAP; beeswarm, waterfall, force plots
- [ ] LIME: local perturbation-based approximation
- [ ] Partial Dependence Plots (1D and 2D PDP)
- [ ] Individual Conditional Expectation (ICE) plots
- [ ] SHAP for images (GradCAM, SHAP DeepLIFT)
- [ ] Regulatory compliance: EU AI Act, GDPR right to explanation

**Checkpoint**: Can you write SQL window functions, handle a 1:100 imbalanced dataset properly, and explain a black-box model to a business stakeholder using SHAP?

---

## Phase 8: Production & MLOps (~3–4 months full-time)

### Module 13 – Model Deployment
- [ ] REST API with FastAPI: path params, request bodies, validation, OpenAPI docs
- [ ] Async endpoints, background tasks, dependency injection
- [ ] Model serialization comparison: pickle, joblib, ONNX, TorchScript
- [ ] Docker: Dockerfile, multi-stage builds, docker-compose, .dockerignore
- [ ] Docker Hub / ECR image registry
- [ ] AWS SageMaker: training jobs, endpoints, batch transform, Pipelines
- [ ] GCP Vertex AI, Azure ML (overview)
- [ ] Hugging Face Spaces: Gradio and Streamlit demos, free GPU
- [ ] NGINX reverse proxy, SSL/TLS certificates (Let's Encrypt), rate limiting
- [ ] A/B testing: statistical significance, sample sizing, multi-armed bandits
- [ ] Model monitoring: data drift (Evidently AI), concept drift, performance dashboards

### Module 14 – MLOps Basics
- [ ] DVC: data versioning, remote storage (S3, GCS), pipelines
- [ ] MLflow: tracking (params, metrics, artifacts), Model Registry, Projects
- [ ] Weights & Biases: runs, sweeps, artifacts, reports
- [ ] CI/CD for ML: GitHub Actions workflows for training, testing, deployment
- [ ] Cookiecutter Data Science: project templates for reproducibility
- [ ] DAG pipelines: Airflow (DAGs, operators, sensors), Kubeflow Pipelines
- [ ] Kubernetes basics: pods, deployments, services, HPA for ML serving
- [ ] Feature stores: Feast (online + offline), Tecton
- [ ] Apache Kafka: producers, consumers, stream processing for real-time ML
- [ ] Apache Spark: DataFrames, MLlib, PySpark for large-scale processing

**Checkpoint**: Can you package a model into a FastAPI Docker container, deploy to AWS, set up MLflow tracking, and write a GitHub Actions CI/CD pipeline?

---

## Phase 9: Projects (~4–6 months full-time)

Work on projects **throughout the journey**, not just at the end.

### Priority Order for AI Engineers
1. **Beginner** (start during Phase 2): House Prices, Titanic, Spam Detection
2. **Intermediate** (start during Phase 5): MNIST, Churn, Fraud Detection
3. **Advanced CV** (during/after Module 11): CIFAR-10, Object Detection, GAN/VAE
4. **Advanced NLP** (during/after Module 12): Sentiment Analysis
5. **Advanced GenAI** (during/after Module 25): LLM Chatbot & RAG System
6. **Advanced MLOps** (during/after Module 14): End-to-End ML Pipeline, Deployment

**Capstone options** (portfolio-grade):
- ML Engineer Capstone: full tabular pipeline + deployment + monitoring
- LLM/RAG Capstone: multi-document RAG + evaluation + production API
- Data Analytics Capstone: SQL + visualization + business story

---

## Phase 10: Advanced Specialized Topics (~2–3 months full-time)

*Treat these as electives — go deep on what excites you most.*

### Module 22 – Reinforcement Learning
- [ ] MDPs: states, actions, rewards, transitions, discount factor
- [ ] Bellman equations; value functions (V* and Q*)
- [ ] Dynamic Programming: policy/value iteration
- [ ] Model-Free RL: Monte Carlo, TD-Learning
- [ ] Q-Learning and tabular Q-tables
- [ ] Deep Q-Networks (DQN): experience replay, target network
- [ ] Double DQN, Dueling DQN, Prioritized Replay
- [ ] Policy Gradient: REINFORCE algorithm, baseline variance reduction
- [ ] Actor-Critic: A2C, A3C, PPO, SAC
- [ ] Multi-Agent RL: cooperative, competitive, mixed settings
- [ ] Environments: OpenAI Gym, PettingZoo, MuJoCo, IsaacGym

### Module 23 – Graph Neural Networks
- [ ] Graph fundamentals: nodes, edges, adjacency matrix, Laplacian
- [ ] Node/link/graph-level tasks
- [ ] Message Passing framework: aggregate, update, readout
- [ ] Graph Convolutional Networks (GCN): Kipf & Welling formulation
- [ ] Graph Attention Networks (GAT): multi-head attention on graphs
- [ ] GraphSAGE: inductive learning via neighborhood sampling
- [ ] Graph Transformers: Graphormer, GT
- [ ] PyTorch Geometric (PyG) and Deep Graph Library (DGL)
- [ ] Applications: molecular property prediction, social network analysis, knowledge graphs

### Module 24 – Audio & Speech Processing
- [ ] Digital audio: sampling rate, bit depth, Fourier Transform, STFT
- [ ] Spectrograms, mel-spectrograms, MFCCs, log-mel features
- [ ] Speech Recognition (ASR): CTC loss, attention-based seq2seq, OpenAI Whisper
- [ ] Text-to-Speech (TTS): Tacotron 2, VITS, neural vocoders (WaveGlow, HiFi-GAN)
- [ ] Voice Cloning: SpeechT5, Tortoise-TTS
- [ ] Audio Classification: CNN on spectrograms, YAMNet, PANNs
- [ ] Music Generation: MusicGen, AudioCraft
- [ ] Voice Activity Detection (VAD); speaker diarization; noise enhancement
- [ ] Libraries: librosa, soundfile, torchaudio, speechbrain

**Checkpoint (Phase 10)**: Pick one domain and implement a non-trivial project: a PPO agent on a custom gym env, a GNN for molecule classification, or an ASR pipeline with Whisper + fine-tuning.

---

## Total Timeline Summary

| Commitment | Minimum | Standard | Comprehensive |
|---|---|---|---|
| **Full-Time (30–40 hrs/week)** | 10–13 months | 15–18 months | 20–22 months |
| **Part-Time (10–15 hrs/week)** | 20–26 months | 28–34 months | 36–40 months |

**Minimum**: Core phases 0–8, skips Phase 10 electives
**Standard**: All 23 modules + 15+ projects + some Phase 10 depth
**Comprehensive**: All modules + all 23 projects + all 3 Phase 10 domains + portfolio capstones

---

## Weekly Study Template

### Full-Time Week (35 hrs)
| Day | Activity | Hours |
|---|---|---|
| Mon | New theory: read module guide, take notes | 7 |
| Tue | Code-along: implement concepts from scratch | 7 |
| Wed | Exercises: solve practice problems | 7 |
| Thu | Project work: apply to current project | 7 |
| Fri | Review + next module preview + Kaggle/Papers | 7 |
| Weekend | Rest or optional extra project time | — |

### Part-Time Week (12 hrs)
| Day | Activity | Hours |
|---|---|---|
| Weekday evenings (×3) | Theory + code-along | 2 hrs × 3 = 6 |
| Saturday | Project work | 4 |
| Sunday | Review + exercises | 2 |

---

## Milestone Badges

Track your progress through these key milestones:

- 🎯 **Phase 0 Complete** — Built a Python project with OOP
- 📊 **Phase 1 Complete** — Built an EDA dashboard in Streamlit
- 🤖 **Phase 2 Complete** — Built and deployed a classification model
- 🌲 **Phase 3 Complete** — Won a Kaggle tabular competition (top 50%)
- 🔍 **Phase 4 Complete** — Segmented customers with unsupervised learning
- 🧠 **Phase 5 Complete** — Trained a neural net in raw PyTorch
- 👁️ **Phase 6 (CV) Complete** — Fine-tuned a YOLO model on custom data
- 💬 **Phase 6 (NLP) Complete** — Fine-tuned BERT for a custom task
- ⏱️ **Branch Complete** — Beat ARIMA with an LSTM forecaster
- 🤖 **Phase 7 Complete** — Built a production RAG system with eval
- 🗄️ **Phase 7.5 Complete** — Written advanced SQL window functions
- 🚀 **Phase 8 Complete** — Deployed a model to AWS with CI/CD
- 🏆 **Phase 9 Complete** — 5+ projects in GitHub portfolio
- 🎓 **Phase 10 Complete** — Advanced domain expertise demonstrated
- 🌟 **AI Engineer Ready** — All phases complete + strong portfolio
