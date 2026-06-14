# Module 08: Unsupervised Learning

**Phase 4 — Unsupervised Learning** | Est. time: 4–6 weeks (full-time) · 8–12 weeks (part-time)

---

## Prerequisites

- Module 07: Feature Engineering

---

## 1. K-Means Clustering

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np

# ─── Elbow method: find optimal K ─────────────────────────────────────────────
inertias, silhouettes = [], []
K_range = range(2, 12)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, "bo-"); ax1.set_title("Elbow Method (Inertia)")
ax2.plot(K_range, silhouettes, "ro-"); ax2.set_title("Silhouette Score (Higher = Better)")
plt.tight_layout(); plt.show()

# ─── Train final model ────────────────────────────────────────────────────────
k_opt = 4   # Choose from elbow/silhouette analysis
km = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)

print(f"Silhouette: {silhouette_score(X_scaled, df['cluster']):.4f}")
print(f"Davies-Bouldin: {davies_bouldin_score(X_scaled, df['cluster']):.4f}")

# ─── Cluster profiles ─────────────────────────────────────────────────────────
print(df.groupby("cluster")[["age", "income", "spend"]].mean())
```

---

## 2. Hierarchical & Density-Based Clustering

```python
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
import hdbscan

# ─── Hierarchical (Ward linkage — usually best) ───────────────────────────────
agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = agg.fit_predict(X_scaled)

# Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(X_scaled, method="ward")
plt.figure(figsize=(14, 6))
dendrogram(linked, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram"); plt.show()

# ─── DBSCAN: density-based, finds arbitrarily shaped clusters ────────────────
# Tune eps with k-distance graph
nn = NearestNeighbors(n_neighbors=4)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])
plt.plot(distances); plt.title("k-Distance Graph (find elbow = eps)"); plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"Clusters: {n_clusters} | Noise points: {n_noise}")

# ─── HDBSCAN: hierarchical DBSCAN, more robust ───────────────────────────────
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels = hdb.fit_predict(X_scaled)
```

---

## 3. Dimensionality Reduction

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap

# ─── PCA: linear, fast, interpretable ────────────────────────────────────────
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# ─── t-SNE: non-linear, for 2D/3D visualization only ─────────────────────────
# Run on PCA-reduced data for speed
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_pca[:, :50])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10)
plt.colorbar(scatter); plt.title("t-SNE Visualization"); plt.show()

# ─── UMAP: faster, better global structure, usable for downstream ML ─────────
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# UMAP for ML (not just visualization — use n_components > 2)
X_umap_50d = umap.UMAP(n_components=50, random_state=42).fit_transform(X_scaled)
```

---

## 4. Anomaly Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# ─── Isolation Forest: fast, scales well ─────────────────────────────────────
iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
labels = iso.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
anomaly_scores = iso.score_samples(X_scaled)  # Lower = more anomalous
print(f"Anomalies detected: {(labels == -1).sum()}")

# ─── LOF: density-based, good for clusters of varying density ────────────────
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X_scaled)
lof_scores = lof.negative_outlier_factor_

# ─── One-Class SVM: train on normal data, detect deviations ──────────────────
ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
ocsvm.fit(X_train_normal)
labels = ocsvm.predict(X_test)   # -1 = anomaly
```

---

## 5. Association Rules (Market Basket Analysis)

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Example: grocery basket data
transactions = [["milk","bread","butter"],["beer","bread"],["milk","bread","beer"],
                ["beer","butter"],["bread","butter","beer","milk"]]

te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_basket = pd.DataFrame(te_array, columns=te.columns_)

# Find frequent itemsets
freq_items = apriori(df_basket, min_support=0.6, use_colnames=True)
print(freq_items)

# Generate rules
rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
rules = rules.sort_values("lift", ascending=False)
print(rules[["antecedents","consequents","support","confidence","lift"]].head(10))
```

---

## Project: Customer Segmentation

**Dataset**: [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) or e-commerce data

Steps:
1. EDA — age, income, spending score distributions
2. Scale features (K-Means is distance-based)
3. Find optimal K — elbow + silhouette
4. Cluster with K-Means and DBSCAN; compare
5. Visualise with UMAP (2D)
6. Profile each segment — what makes them different?
7. Provide actionable business recommendations

**[← Module 07](../07-feature-engineering/README.md)** | **[→ Module 09](../09-neural-networks-basics/README.md)**
