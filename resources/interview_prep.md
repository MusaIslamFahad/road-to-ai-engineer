# AI Engineer Interview Preparation Guide

A comprehensive guide covering every interview category you'll face as a Generalist AI Engineer.

---

## Interview Types Overview

| Round | Format | Focus |
|---|---|---|
| **Recruiter Screen** | 30 min call | Background, motivations, logistics |
| **Technical Phone Screen** | 45–60 min | 1–2 coding problems + ML basics |
| **ML Breadth** | 60 min | ML concepts across all domains |
| **ML Depth** | 60 min | Deep dive in your specialization |
| **Coding / DSA** | 60 min | Algorithms and data structures |
| **ML System Design** | 60 min | Design a large-scale AI system |
| **Behavioral / Leadership** | 45–60 min | STAR stories, culture fit |

---

## Part 1: ML Fundamentals

### Supervised Learning
**Q: What is the bias-variance tradeoff?**
> *High bias* (underfitting): model too simple, can't capture patterns. *High variance* (overfitting): model memorizes training data, fails on new data. You tune complexity (tree depth, regularization, number of neurons) to find the sweet spot. Cross-validation is how you measure it; learning curves help diagnose it.

**Q: When would you use L1 vs. L2 regularization?**
> *L2 (Ridge)*: shrinks all coefficients toward zero, rarely sets them to zero. Better when all features are potentially relevant. *L1 (Lasso)*: drives some coefficients to exactly zero — automatic feature selection. Use when you suspect many features are irrelevant. *ElasticNet*: combines both; use for correlated features.

**Q: How does a Random Forest differ from a single Decision Tree?**
> Random Forest builds many trees on bootstrap samples (bagging) and takes a majority vote. Two sources of randomness: bootstrap sampling of rows AND random subset of features at each split. This decorrelates the trees so averaging reduces variance without increasing bias. A single tree has low bias but high variance.

**Q: Explain XGBoost vs. Random Forest.**
> Both are tree ensembles but built differently. Random Forest builds trees in parallel on bootstrap samples and averages. XGBoost builds trees *sequentially*, each one correcting the residual errors of the previous. XGBoost also adds L1/L2 regularization, handles missing values natively, and is generally more accurate but slower to tune.

**Q: What is data leakage and how do you detect and prevent it?**
> Leakage occurs when information from the target (or the future) is accidentally included in training features. Detection: suspiciously high accuracy, a feature that strongly correlates with the target but shouldn't logically. Prevention: always split first, then apply all preprocessing inside a Pipeline fitted only on train data. Be especially careful with time series (no future data), group splits (no patient overlap in medical), and target statistics (use out-of-fold encoding).

**Q: How do you handle class imbalance?**
> Multiple complementary strategies: (1) Resampling: SMOTE for oversampling minority, or random undersampling — always inside CV folds. (2) Algorithm-level: `class_weight='balanced'`, `scale_pos_weight` in XGBoost. (3) Threshold tuning: optimize threshold on val set for your business metric (F1, recall). (4) Evaluation: use PR-AUC or F1 macro, not accuracy.

**Q: Walk me through cross-validation. When would you use each type?**
> *K-Fold*: split data into K folds, train on K-1, test on 1, repeat K times. Use for i.i.d. data. *Stratified K-Fold*: preserve class ratio in each fold — always use for classification. *Time Series Split*: train on past, validate on future (walk-forward) — use for any temporal data. *Group K-Fold*: ensure same group (patient, user) never appears in both train and test.

---

## Part 2: Deep Learning

**Q: Explain backpropagation step by step.**
> 1. Forward pass: input flows through layers; record all intermediate activations. 2. Compute loss between prediction and ground truth. 3. Backward pass: apply chain rule from loss back through each layer — compute ∂L/∂W for every weight. 4. Update weights: W ← W − lr × ∂L/∂W. Key insight: the chain rule means gradients at earlier layers are products of all later layer gradients — this is why deep networks suffer vanishing gradients with sigmoid/tanh.

**Q: Why do we need batch normalization?**
> It normalizes the activations of each layer to have zero mean and unit variance (then learned scale/shift γ, β). Benefits: (1) Reduces internal covariate shift, so deeper layers don't have to adapt to shifting input distributions. (2) Allows higher learning rates. (3) Acts as a mild regularizer (reduces need for Dropout). (4) Reduces sensitivity to weight initialization.

**Q: Compare Adam, SGD with Momentum, and AdamW. When do you use each?**
> *SGD+Momentum*: simple, generalizes well for vision models, but requires careful LR tuning. *Adam*: adaptive per-parameter learning rates + momentum; converges fast; default choice. *AdamW*: Adam with decoupled weight decay (L2 applied to weights, not gradients) — better regularization; recommended for transformers and modern models. Rule of thumb: Adam/AdamW for transformers and quick experiments; SGD+Momentum for CV when final performance matters.

**Q: What are vanishing and exploding gradients, and how do you fix them?**
> *Vanishing*: gradients become exponentially small in early layers, preventing learning. Cause: many sigmoid/tanh activations in deep networks; bad initialization. Fixes: ReLU/GELU activations, residual connections (ResNets), layer normalization, careful initialization (Xavier/He). *Exploding*: gradients become huge, causing unstable training. Fix: gradient clipping (`clip_grad_norm_`).

**Q: When would you use a CNN vs. RNN vs. Transformer?**
> *CNN*: spatial locality matters, translation invariance is useful — images, audio spectrograms, some 1D signals. *RNN/LSTM*: sequential data with long-range dependencies, but now largely replaced by transformers. *Transformer*: any sequence task where long-range dependencies matter — NLP, vision (ViT), time series. Transformers dominate now; use LSTMs only for resource-constrained deployment.

---

## Part 3: Computer Vision

**Q: How does a convolutional layer work?**
> A learnable filter (e.g., 3×3) slides over the input spatial dimensions, computing dot products at each position — this is convolution. Key parameters: kernel size (receptive field), stride (step size), padding (preserve spatial dims), number of filters (output channels). Each filter learns to detect a different local pattern. Deeper layers combine these into more abstract features.

**Q: Explain transfer learning. When do you fine-tune vs. freeze the backbone?**
> Transfer learning initializes weights from a model trained on a large dataset (ImageNet) and adapts it. *Freeze + train head*: use when your dataset is small or similar to the source domain. Fast, low overfisk. *Fine-tune all layers*: use when you have more data or the target domain is quite different. Unfreeze later layers first, then earlier layers with a smaller learning rate (differential LR).

**Q: How does YOLO work for object detection?**
> YOLO divides the image into an S×S grid. Each grid cell predicts B bounding boxes (center x/y, width, height, objectness score) and C class probabilities simultaneously in one forward pass. Post-processing: apply NMS to remove duplicate boxes. This makes it extremely fast vs. two-stage detectors like Faster R-CNN that first propose regions then classify them.

**Q: What is the difference between semantic and instance segmentation?**
> *Semantic*: assigns a class label to every pixel; doesn't distinguish between instances of the same class. All cars are "car". *Instance*: detects and segments individual objects; two cars get different masks. Semantic = pixel classification; instance = object detection + mask prediction (Mask R-CNN). *Panoptic* combines both.

---

## Part 4: NLP & Transformers

**Q: Explain self-attention from scratch.**
> Each token creates three vectors from its embedding: Query (Q), Key (K), Value (V) via learned linear projections. Attention score between token i and token j = softmax(Qᵢ · Kⱼ / √d_k). The output for token i is the weighted sum of all Value vectors. Multi-head: run H attention heads in parallel, concatenate and project. This lets each token attend to all other tokens regardless of distance — no recurrence needed.

**Q: What is the difference between BERT and GPT?**
> *BERT*: encoder-only. Bidirectional — sees full context (left and right). Pre-trained with Masked LM (predict [MASK] tokens) and Next Sentence Prediction. Best for: classification, NER, QA (extractive). *GPT*: decoder-only. Unidirectional (left to right only, causal attention mask). Pre-trained with Causal LM (predict next token). Best for: text generation, in-context learning, instruction following.

**Q: When would you choose fine-tuning vs. RAG vs. prompt engineering?**
> *Prompt engineering*: no data, need quick results, task is well-served by zero/few-shot. Cost: none. *RAG*: need access to a large, frequently updated knowledge base. Model doesn't need to memorize — retrieves at inference. Best for Q&A over documents. *Fine-tuning*: need to change model behavior, style, or format; task-specific knowledge; consistent tone; few-shot isn't enough. More expensive — requires data and compute.

**Q: What are the failure modes of RAG systems?**
> (1) *Retrieval failure*: relevant document not retrieved — fix with hybrid search (dense + BM25), better chunking, query expansion. (2) *Context overload*: retrieved context is too long or noisy — fix with re-ranking, contextual compression. (3) *Hallucination despite retrieval*: LLM ignores context or confabulates — fix with better prompting, faithfulness evaluation (RAGAS). (4) *Chunking mismatch*: answers span chunk boundaries — fix with semantic chunking or parent-child chunks.

---

## Part 5: MLOps & Production

**Q: How do you monitor a ML model in production?**
> Three levels: (1) *Infrastructure*: CPU/GPU utilization, latency, throughput, error rate — Prometheus + Grafana. (2) *Data quality*: input distribution drift (PSI, KS-test), missing values, schema violations — Evidently AI. (3) *Model performance*: accuracy, precision, recall tracked against ground truth labels when available; proxy metrics when not. Set thresholds and alert via PagerDuty/Slack. Schedule automatic retraining when drift exceeds threshold.

**Q: Walk me through your CI/CD pipeline for ML.**
> 1. Developer pushes to feature branch → PR opened. 2. GitHub Actions triggers: run unit tests (pytest), data validation tests, lint (black, isort). 3. If tests pass: run training job on small data slice to verify pipeline end-to-end. 4. Merge to main → full training job → model registered in MLflow. 5. Staging deploy → integration tests + shadow traffic. 6. Manual approval → production deploy with canary release (5% → 25% → 100%). 7. Monitor for 24h; rollback if metrics regress.

**Q: What is A/B testing? When would you use a multi-armed bandit instead?**
> *A/B test*: split traffic 50/50, collect data for N days (based on power analysis), run a statistical test (z-test for proportions). Clean, interpretable, but wasteful — sends half of traffic to possibly the loser. *Multi-armed bandit* (Thompson Sampling): continuously updates allocation toward the better model; exploits while still exploring. Use MAB when: faster iteration needed, business cost of serving the loser is high, or experiment duration is flexible.

---

## Part 6: ML System Design

### Framework: RADARS

**R** — Requirements clarification (functional + non-functional)
**A** — Architecture overview (components and data flow)
**D** — Data: collection, storage, feature engineering, pipeline
**A** — Algorithm selection and training strategy
**R** — Real-time vs. batch serving; latency / throughput targets
**S** — Scale, monitoring, failure modes, iteration plan

---

### Common Design Questions

#### "Design a real-time fraud detection system"

**Requirements**: <100ms latency, 10K transactions/sec, ~0.1% fraud rate

**Architecture**:
```
Transaction → Kafka → Feature Service → ML Model API → Decision
                ↓
        Feature Store (Redis - online)
        Historical Store (BigQuery - offline training)
```

**Model**: LightGBM (fast inference) + neural embedding for behavioral sequences. Two-stage: fast rule-based filter → ML model.

**Features**: transaction amount, merchant category, user history (avg spend, frequency), device fingerprint, velocity features (txns in last 1h/24h).

**Training**: Weekly retrain on 90-day window. Walk-forward validation. Monitor PSI daily.

**Monitoring**: alert if fraud rate spikes, drift in feature distributions, P99 latency >80ms.

---

#### "Design a document Q&A system (RAG)"

**Components**:
```
Documents → Ingestion Pipeline → Chunker → Embedder → Vector DB
User Query → Query Processor → Retriever → Re-ranker → LLM → Response
```

**Chunking**: recursive character splitting (512 tokens, 50 overlap). For PDFs: preserve paragraph boundaries.

**Retrieval**: hybrid (FAISS dense + BM25 sparse), combine with RRF. Top-20 candidates.

**Re-ranking**: cross-encoder (ms-marco-MiniLM) to top-5.

**LLM**: GPT-4o for accuracy, fall back to GPT-4o-mini for cost. Streaming response.

**Evaluation**: RAGAS (faithfulness, answer relevancy, context recall) on golden Q&A set.

**Scaling**: async ingestion with Celery + Redis. Batch embedding with OpenAI. Pinecone for vector search at scale.

---

#### "Design a recommendation system for 10M users"

**Two-stage architecture** (industry standard):
1. **Candidate generation** (recall): fast approximate methods — matrix factorization (ALS), two-tower model, item-item collaborative filtering. Output: top-500 candidates from millions.
2. **Ranking** (precision): LightGBM or a small DNN with rich cross features. Output: top-10 sorted.

**Features**: user history, item metadata, context (time, device, location), cross features.

**Training**: implicit feedback (clicks, watches, purchases). Negative sampling: 1:10 ratio.

**Serving**: candidate generation offline (nightly), ranking online (<50ms). Redis for feature serving.

**Cold start**: content-based for new items; demographic-based for new users.

---

## Part 7: Coding / DSA for ML Engineers

ML interviews often include 1–2 LeetCode-style problems. Focus on:

| Category | Priority | Key Problems |
|---|---|---|
| Arrays & Hashing | High | Two Sum, Sliding Window, Prefix Sum |
| Binary Search | High | Search in Rotated Array, Find Peak |
| Two Pointers | High | Merge Intervals, Container With Most Water |
| Linked Lists | Medium | Reverse, Detect Cycle |
| Trees | Medium | BFS/DFS, Lowest Common Ancestor |
| Dynamic Programming | Medium | Coin Change, Longest Subsequences |
| Sorting | High | Know built-in sort, understand merge sort |
| Graphs | Medium | BFS/DFS, Dijkstra |

**ML-specific coding questions**:
- Implement gradient descent from scratch
- Implement k-means clustering
- Compute cosine similarity without using scipy
- Implement a simple tokenizer (BPE basics)
- Write a train/val/test split that respects time order

---

## Part 8: Behavioral Questions (STAR Format)

**Situation / Task / Action / Result**

Common questions and what they're really asking:

| Question | What They Want to Hear |
|---|---|
| "Tell me about a time you improved a model significantly" | Problem framing, debugging skills, iteration speed |
| "Describe a project where data quality was a challenge" | Data intuition, pragmatic solutions |
| "Tell me about a time you disagreed with a teammate" | Communication, humility, collaborative problem solving |
| "Describe a failed project. What did you learn?" | Self-awareness, learning mindset |
| "How do you stay current with ML research?" | Curiosity, learning habits |
| "Tell me about the most complex system you've built" | Scope of work, technical depth, ownership |

**Prepare 5–7 STAR stories** that can flex to cover multiple question types.

---

## Part 9: Questions to Ask the Interviewer

- "What does the ML infrastructure look like — is there a platform team or do engineers own their own infra?"
- "How do you evaluate and adopt new ML techniques (e.g., GenAI) in the team's work?"
- "What's the ratio of time spent on model development vs. data engineering vs. deployment?"
- "What does the model monitoring story look like today? What's the biggest gap?"
- "What would success look like in the first 6 months for this role?"

---

## Resources for Interview Prep

| Resource | Use |
|---|---|
| [Ace the Data Science Interview](https://www.acethedatascienceinterview.com/) | 201 ML interview questions |
| [LeetCode](https://leetcode.com/explore/learn/) | DSA practice |
| [ML System Design](https://www.educative.io/blog/ml-system-design) | System design patterns |
| [Chip Huyen's ML Interviews Book](https://huyenchip.com/ml-interviews-book/) | Free — comprehensive ML interview guide |
| [Deep Learning Interviews](https://arxiv.org/abs/2201.00650) | 200+ DL questions |
| [Glassdoor / Levels.fyi](https://www.levels.fyi/) | Company-specific questions and compensation |
