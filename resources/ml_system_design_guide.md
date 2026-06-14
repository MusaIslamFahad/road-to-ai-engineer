# ML System Design Guide

How to design scalable, production-ready ML systems — a framework for design interviews and real engineering.

---

## The RADARS Framework

Use this for any ML system design question:

**R** — Requirements (functional + non-functional)  
**A** — Architecture overview  
**D** — Data (collection, storage, pipeline, features)  
**A** — Algorithm (model choice, training strategy)  
**R** — Real-time serving (latency, throughput, SLA)  
**S** — Scale, monitoring, iteration, failure handling  

---

## Part 1: Worked Example — Real-Time Fraud Detection

### R — Requirements

**Functional**
- Score every transaction within 100ms of it being initiated
- Flag transactions above a fraud probability threshold
- Support review workflow for flagged transactions

**Non-Functional**
- Throughput: 10,000 transactions/second peak
- Latency: p99 < 100ms end-to-end
- Availability: 99.99% (4 nines)
- False positive rate < 0.5% (customer experience)
- Recall > 90% on known fraud patterns

---

### A — Architecture

```
Transaction API
     ↓
Kafka Topic "raw-transactions"
     ↓                    ↓
Feature Service     Rule Engine (fast, high-recall pre-filter)
(Redis lookup)           ↓ (only suspicious transactions)
     ↓
ML Scoring Service (FastAPI + ONNX model)
     ↓
Decision Engine (threshold + business rules)
     ↓              ↓
Allow         Flag/Block → Alert Queue → Human Review
     ↓
Kafka Topic "scored-transactions" → Data Warehouse (for retraining)
```

---

### D — Data

**Training data**: 90 days of labeled transactions (fraud label applied retrospectively by dispute team)

**Features**:
```python
# Transaction-level
- amount, merchant_category, merchant_country
- time_of_day, day_of_week, is_weekend

# User velocity features (computed in real-time from Redis)
- txn_count_1h, txn_count_24h
- amount_sum_1h, amount_sum_24h
- distinct_merchants_24h
- failed_txn_count_24h

# User profile features (from offline feature store)
- avg_txn_amount_30d, home_country, device_fingerprint_age
- account_age_days, historical_dispute_rate

# Contextual
- is_international, currency_mismatch, amount_vs_user_avg_ratio
```

**Feature store architecture**:
- **Online store** (Redis): velocity features computed per transaction, TTL-keyed by user_id + time window
- **Offline store** (BigQuery): user profile features, updated nightly

---

### A — Algorithm

**Model choice**: Two-stage
1. **Stage 1 — Rule engine** (< 1ms): simple heuristics to catch obvious fraud and pre-filter. Cheap, explainable, high recall.
2. **Stage 2 — LightGBM** (5-10ms): trained on labeled data with SMOTE + `is_unbalance=True`. Handles imbalance (0.1% fraud rate).

**Why LightGBM over neural network?**
- Tabular data with engineered features → LightGBM wins
- Interpretable with SHAP (regulatory requirement)
- Fast inference (< 10ms on CPU via ONNX)
- Handles missing features gracefully

**Training strategy**:
- Walk-forward validation (predict next week from past 8 weeks)
- Negative sampling: 1 fraud : 20 non-fraud in training batches
- Class weight: `scale_pos_weight = 200` (neg/pos ratio)
- Retrain weekly on rolling 90-day window

---

### R — Real-Time Serving

```python
# FastAPI scoring service
from fastapi import FastAPI
import onnxruntime as ort
import redis, json, numpy as np

app = FastAPI()
session = ort.InferenceSession("fraud_model.onnx",
    providers=["CPUExecutionProvider"])
redis_client = redis.Redis(host="redis", port=6379)

@app.post("/score")
async def score_transaction(txn: Transaction):
    # 1. Fetch velocity features from Redis (< 1ms)
    key = f"user:{txn.user_id}:velocity"
    velocity = json.loads(redis_client.get(key) or "{}")

    # 2. Build feature vector
    features = build_features(txn, velocity)
    X = np.array(features, dtype=np.float32).reshape(1, -1)

    # 3. ONNX inference (5-10ms)
    fraud_prob = session.run(None, {"input": X})[0][0][1]

    # 4. Update velocity in Redis (async)
    update_velocity(txn, redis_client)

    return {"fraud_probability": float(fraud_prob),
            "decision": "block" if fraud_prob > 0.85 else
                        "review" if fraud_prob > 0.5 else "allow"}
```

**Latency budget**:
- Kafka consume: ~5ms
- Redis lookup: ~1ms
- Feature assembly: ~2ms
- ONNX inference: ~8ms
- Decision + Kafka produce: ~5ms
- **Total: ~21ms (well within 100ms SLA)**

---

### S — Scale & Monitoring

**Scaling**:
- Scoring service: 10 replicas behind load balancer, HPA on CPU > 70%
- Redis: Redis Cluster with 3 primary + 3 replica nodes
- Kafka: 12 partitions for raw-transactions topic, 1 consumer group per scoring service pod

**Monitoring**:
- Business: fraud rate, false positive rate, daily loss prevented
- Model: score distribution drift (PSI), SHAP feature importance drift
- Infrastructure: p50/p95/p99 latency, error rate, throughput per second
- Alerts: fraud rate spikes 3σ, model latency p99 > 80ms, feature drift PSI > 0.2

**Retraining trigger**: Automated weekly + on-demand if PSI > 0.2 on key features

---

## Part 2: Worked Example — Recommendation System

### Requirements
- 10M active users, 5M items
- Personalized home feed recommendations (top-20 items)
- < 50ms serving latency
- A/B test new models before full rollout

### Two-Stage Architecture

```
User Request
     ↓
[Stage 1: Candidate Generation] — Recall (speed)
     ├── Two-Tower Neural Network (ANN lookup, ~1ms) → top 500
     ├── Collaborative Filtering (ALS) → top 500
     └── Item-Item similarity → top 200
     ↓
Merge + deduplicate → ~800 candidates
     ↓
[Stage 2: Ranking] — Precision (quality)
     LightGBM ranker with cross features
     (user_id, item_id, context features)
     → Ranked top-20
     ↓
[Post-Processing]
     └── Diversity injection, freshness boost, business rules
     ↓
Final top-20 → User
```

### Feature Engineering for Recommendations

```python
# User features (from offline feature store, refreshed daily)
user_features = {
    "age_bucket", "gender", "country",
    "avg_session_length_7d", "active_days_30d",
    "top_categories_30d",           # Multi-hot encoded
    "last_interacted_item_emb",     # 64-dim embedding
}

# Item features
item_features = {
    "category", "subcategory", "price_bucket",
    "avg_rating", "n_reviews_log",
    "days_since_published",
    "item_emb",                     # 64-dim embedding from item2vec
}

# Cross / interaction features (computed at ranking time)
cross_features = {
    "user_category_affinity",       # P(user clicked this category historically)
    "item_novelty_for_user",        # Is this item new to the user?
    "time_since_last_session",
    "similar_items_clicked_7d",
}
```

### Cold Start Handling

| Scenario | Strategy |
|---|---|
| New user | Demographic-based, top-popular per country/age |
| New item | Content-based (similar to user's history by item metadata) |
| New user + new item | Contextual bandits: explore efficiently |

---

## Part 3: Common Design Questions

### "Design a Search Ranking System"

```
Query → Query Understanding (spell check, intent, entity extraction)
     ↓
Retrieval (inverted index, BM25) → top 1000 candidates
     ↓
Feature extraction per (query, document) pair
     ↓
Pointwise or Listwise Learning-to-Rank (LightGBM or BERT cross-encoder)
     ↓
Re-ranking with business rules (sponsored, freshness, safety)
     ↓
Top-10 results
```

### "Design a Content Moderation System"

```
Content Upload
     ↓
[Tier 1: Hash Matching] — Known harmful content (< 1ms)
     ↓ (if no match)
[Tier 2: Fast Classifier] — CNN or text classifier (< 50ms)
     ↓ (if borderline)
[Tier 3: Heavy Model] — Large multimodal model (< 2s)
     ↓ (if still uncertain)
Human Review Queue
```

### "Design a Real-Time Bidding ML System"

- **SLA**: 10ms from bid request to response
- **Volume**: 500K requests/second
- **Model**: Logistic regression (ultra-fast) or LightGBM with ONNX
- **Features**: user vector (pre-computed), ad vector (pre-computed), context
- **Serving**: 200+ geographically distributed inference nodes

---

## Part 4: Key Trade-offs to Discuss

| Decision | Option A | Option B | When to choose A |
|---|---|---|---|
| Model complexity | Simple (logistic, linear) | Complex (deep learning) | Latency SLA < 20ms |
| Feature freshness | Real-time (streaming) | Batch (daily) | User behaviour changes rapidly |
| Serving | Online (request-time) | Batch (pre-computed) | Millions of users, same recommendations |
| Architecture | Monolith | Microservices | Small team, early stage |
| Storage | Redis | PostgreSQL | Need sub-millisecond feature lookup |
| Retraining | Trigger-based | Scheduled | Production drift unpredictable |

---

## Part 5: Estimation Framework

Always do quick math to check feasibility:

```
Fraud detection example:
- 10,000 TPS peak × 86,400 sec/day = 864M transactions/day
- Each transaction: 200 bytes → 172 GB/day of raw data
- Feature vector: 100 floats × 4 bytes = 400 bytes
- Model inference: 8ms → max 125 predictions/sec per CPU core
- To handle 10K TPS: 10,000 / 125 = 80 CPU cores → ~8 pods with 10 cores each

Storage for 90-day training window:
- 864M txn/day × 90 days × 400 bytes = ~31 TB → use BigQuery (columnar, cheap)
```

---

## Interview Tips

1. **Clarify requirements first** — ask: "Is this real-time or batch?", "What's the latency SLA?", "How much data?"
2. **Start simple** — propose a baseline (logistic regression, collaborative filtering) before jumping to transformers
3. **Think in layers** — candidate generation (recall) → ranking (precision) → post-processing (business rules)
4. **Quantify everything** — traffic, storage, latency, costs
5. **Discuss trade-offs** — show you understand the why, not just the what
6. **Mention monitoring** — always close with "how do we know it's working in production?"
