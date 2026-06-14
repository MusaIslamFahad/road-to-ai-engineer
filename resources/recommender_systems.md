# Recommender Systems Guide

Complete reference for building recommendation systems — from collaborative filtering to modern deep learning approaches.

---

## System Types

| Type | How It Works | Cold Start | Needs Labels | Example |
|---|---|---|---|---|
| **Collaborative Filtering** | Similar users/items | ❌ Problem | Implicit/explicit | Netflix, Spotify |
| **Content-Based** | Item features | ✅ OK | No | News recommenders |
| **Hybrid** | Both combined | ✅ Better | Mixed | Most production |
| **Knowledge-Based** | Rules + constraints | ✅ Great | No | Travel booking |
| **Deep Learning** | Learned embeddings | Partial | Implicit | YouTube, TikTok |

---

## Part 1: Collaborative Filtering

### User-Based CF

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Build user-item matrix
ratings = pd.DataFrame({
    "user":   [1,1,1,2,2,3,3,3,4,4],
    "movie":  ["A","B","C","A","C","B","C","D","A","D"],
    "rating": [5,3,4,4,2,5,4,3,3,5]
})
matrix = ratings.pivot(index="user", columns="movie", values="rating").fillna(0)

# User similarity (cosine)
user_sim = pd.DataFrame(
    cosine_similarity(matrix),
    index=matrix.index, columns=matrix.index
)

def predict_rating(user_id, item, matrix, user_sim, k=3):
    """Predict rating for user on item using top-k similar users."""
    similar = user_sim[user_id].drop(user_id).sort_values(ascending=False)
    top_k   = similar.head(k)
    # Only keep users who rated this item
    top_k   = top_k[top_k.index.isin(matrix.index[matrix[item] > 0])]
    if top_k.empty: return matrix[item].mean()
    weights = top_k.values
    ratings = matrix.loc[top_k.index, item].values
    return np.dot(weights, ratings) / (np.sum(np.abs(weights)) + 1e-8)

# Recommend top N unseen items for a user
def recommend(user_id, matrix, user_sim, n=5):
    seen     = set(matrix.columns[matrix.loc[user_id] > 0])
    unseen   = [col for col in matrix.columns if col not in seen]
    scores   = {item: predict_rating(user_id, item, matrix, user_sim) for item in unseen}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
```

### Item-Based CF

```python
# Item similarity (cosine)
item_sim = pd.DataFrame(
    cosine_similarity(matrix.T),
    index=matrix.columns, columns=matrix.columns
)

def predict_item_based(user_id, item, matrix, item_sim, k=5):
    """Predict rating using top-k similar items the user has rated."""
    user_ratings = matrix.loc[user_id]
    rated_items  = user_ratings[user_ratings > 0].index.tolist()
    if item in rated_items: rated_items.remove(item)
    if not rated_items: return user_ratings.mean()

    sims  = item_sim[item][rated_items].sort_values(ascending=False).head(k)
    rates = user_ratings[sims.index]
    return np.dot(sims.values, rates.values) / (np.sum(np.abs(sims.values)) + 1e-8)
```

---

## Part 2: Matrix Factorisation

### SVD (scipy)

```python
from scipy.sparse.linalg import svds
import numpy as np

# Fill NaN with global mean
matrix_filled = matrix.fillna(matrix.stack().mean())
U, sigma, Vt = svds(matrix_filled.values, k=50)   # k = latent factors
sigma_diag = np.diag(sigma)

# Reconstruct full rating matrix
predicted = pd.DataFrame(
    U @ sigma_diag @ Vt,
    index=matrix.index, columns=matrix.columns
)

def recommend_svd(user_id, matrix, predicted, n=10):
    seen  = set(matrix.columns[matrix.loc[user_id].notna()])
    preds = predicted.loc[user_id].drop(index=list(seen))
    return preds.sort_values(ascending=False).head(n)
```

### Alternating Least Squares (ALS) with Implicit

```python
import implicit
import scipy.sparse as sparse
import numpy as np

# Build user-item sparse matrix (implicit: purchase counts, clicks, plays)
user_ids  = [0, 0, 1, 1, 2, 2, 2]
item_ids  = [0, 1, 1, 2, 0, 2, 3]
data      = [1, 2, 1, 3, 1, 4, 2]  # counts
user_item = sparse.csr_matrix((data, (user_ids, item_ids)), shape=(3, 4))

# Train ALS
model = implicit.als.AlternatingLeastSquares(
    factors=50,           # number of latent factors
    regularization=0.01,
    iterations=20
)
model.fit(user_item)

# Recommend for user 0
ids, scores = model.recommend(0, user_item[0], N=10, filter_already_liked_items=True)
print(list(zip(ids, scores)))

# Similar items to item 2
sim_ids, sim_scores = model.similar_items(2, N=5)
```

---

## Part 3: Content-Based Filtering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

movies = pd.DataFrame({
    "title":       ["Inception", "Interstellar", "The Dark Knight", "Arrival"],
    "description": [
        "dream heist thriller mind sci-fi",
        "space time travel sci-fi emotional",
        "batman joker crime thriller dark",
        "alien language time travel sci-fi"
    ]
})

# TF-IDF on descriptions
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["description"])

# Cosine similarity between all pairs
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_recommend(title, movies, cos_sim, n=3):
    idx      = movies[movies["title"] == title].index[0]
    scores   = list(enumerate(cos_sim[idx]))
    scores   = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [movies.iloc[i]["title"] for i, _ in scores]

print(content_recommend("Inception", movies, cos_sim))
# → ['Interstellar', 'Arrival', 'The Dark Knight']
```

---

## Part 4: Neural Collaborative Filtering (NCF)

```python
import torch
import torch.nn as nn

class NCF(nn.Module):
    """Neural Collaborative Filtering: GMF + MLP combined."""

    def __init__(self, n_users, n_items, emb_dim=64, hidden_dims=[256, 128, 64]):
        super().__init__()
        # Generalised Matrix Factorisation (GMF) embeddings
        self.user_emb_gmf = nn.Embedding(n_users, emb_dim)
        self.item_emb_gmf = nn.Embedding(n_items, emb_dim)

        # MLP embeddings
        self.user_emb_mlp = nn.Embedding(n_users, emb_dim)
        self.item_emb_mlp = nn.Embedding(n_items, emb_dim)

        # MLP layers
        mlp_layers = []
        in_dim = emb_dim * 2
        for h in hidden_dims:
            mlp_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        self.mlp = nn.Sequential(*mlp_layers)

        # Final output layer
        self.output = nn.Linear(emb_dim + hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # GMF path: element-wise product
        u_gmf = self.user_emb_gmf(user_ids)
        i_gmf = self.item_emb_gmf(item_ids)
        gmf_out = u_gmf * i_gmf

        # MLP path
        u_mlp = self.user_emb_mlp(user_ids)
        i_mlp = self.item_emb_mlp(item_ids)
        mlp_out = self.mlp(torch.cat([u_mlp, i_mlp], dim=-1))

        # Concatenate and predict
        out = self.output(torch.cat([gmf_out, mlp_out], dim=-1))
        return self.sigmoid(out).squeeze()

# Training with implicit negative sampling
def train_ncf(model, train_df, n_items, epochs=10, neg_ratio=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _, row in train_df.iterrows():
            user = torch.tensor([row["user_id"]])
            item = torch.tensor([row["item_id"]])
            # Positive sample
            pos_pred = model(user, item)
            pos_loss  = criterion(pos_pred, torch.ones(1))
            # Negative samples
            neg_items = torch.randint(0, n_items, (neg_ratio,))
            neg_preds = model(user.repeat(neg_ratio), neg_items)
            neg_loss  = criterion(neg_preds, torch.zeros(neg_ratio))
            loss = pos_loss + neg_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
```

---

## Part 5: Two-Tower Model (Production Scale)

```python
class TwoTowerModel(nn.Module):
    """
    Industry-standard architecture for large-scale candidate retrieval.
    User tower and item tower trained together via contrastive loss.
    At serving time: pre-compute item embeddings → ANN lookup.
    """

    def __init__(self, n_users, n_items, n_user_features, n_item_features, emb_dim=128):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Embedding(n_users, emb_dim),
            nn.Linear(emb_dim + n_user_features, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Embedding(n_items, emb_dim),
            nn.Linear(emb_dim + n_item_features, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, user_ids, user_feats, item_ids, item_feats):
        u_emb = self.user_tower[0].weight[user_ids]  # embedding lookup simplification
        i_emb = self.item_tower[0].weight[item_ids]
        # Dot product similarity
        return (u_emb * i_emb).sum(dim=-1)
```

---

## Part 6: Evaluation Metrics

```python
import numpy as np

def precision_at_k(recommended, relevant, k):
    """Fraction of top-k recommendations that are relevant."""
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items in top-k recommendations."""
    if not relevant: return 0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    """Normalised Discounted Cumulative Gain — rewards relevant items ranked higher."""
    dcg  = sum(1/np.log2(i+2) for i,r in enumerate(recommended[:k]) if r in set(relevant))
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def mean_average_precision(all_recommended, all_relevant, k):
    """MAP@K over all users."""
    aps = []
    for rec, rel in zip(all_recommended, all_relevant):
        rel_set = set(rel)
        hits    = 0
        precisions = []
        for i, item in enumerate(rec[:k]):
            if item in rel_set:
                hits += 1
                precisions.append(hits / (i+1))
        aps.append(np.mean(precisions) if precisions else 0)
    return np.mean(aps)

# Example evaluation
recommended = [["A","B","C","D","E"], ["B","C","D","E","F"]]
relevant     = [["A","C"],             ["C","E"]]
print(f"P@3:  {np.mean([precision_at_k(r,rel,3) for r,rel in zip(recommended,relevant)]):.4f}")
print(f"R@5:  {np.mean([recall_at_k(r,rel,5)    for r,rel in zip(recommended,relevant)]):.4f}")
print(f"NDCG@5: {np.mean([ndcg_at_k(r,rel,5)   for r,rel in zip(recommended,relevant)]):.4f}")
print(f"MAP@5: {mean_average_precision(recommended, relevant, 5):.4f}")
```

---

## Cold Start Strategies

| Scenario | Strategy |
|---|---|
| New user, no history | Ask for preferences onboarding; popularity-based; demographic-based |
| New item, no ratings | Content-based similarity to existing items; feature-based |
| New user + new item | Contextual bandits (Thompson Sampling) for efficient exploration |
| Sparse user (<5 ratings) | Hybrid: 80% content-based + 20% CF |

---

## Production Architecture

```
Offline (nightly):
  Training data → Two-Tower model → Pre-compute all item embeddings
  → Store in vector DB (Faiss, Pinecone)

Online (< 50ms):
  User request
       ↓
  [Stage 1: Candidate Generation]
    User embedding (lookup) → ANN search → top 500 candidates
       ↓
  [Stage 2: Ranking]
    (user, item) feature pairs → LightGBM ranker → top 20
       ↓
  [Post-Processing]
    Diversity injection + freshness boost + business rules
       ↓
  Return top 20 to user
```
