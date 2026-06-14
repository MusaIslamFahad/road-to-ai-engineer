# Module 23: Graph Neural Networks

**Phase 10 — Advanced Specialized Topics** | Est. time: 1–1.5 months (full-time) · 2–3 months (part-time)

---

## Learning Objectives

By the end of this module, you will:
- Represent real-world data as graphs and solve graph-level tasks
- Implement GCN, GAT, and GraphSAGE from scratch
- Use PyTorch Geometric for production GNN workloads
- Apply GNNs to social networks, molecular data, and recommendation systems

---

## Prerequisites

- Module 10: Deep Learning Frameworks (PyTorch)
- Basic linear algebra (Module 00)

---

## Topics Covered

### Graph Fundamentals
- Graphs: nodes (vertices), edges, directed/undirected, weighted
- Adjacency matrix, degree matrix, Laplacian matrix
- Graph types: homogeneous, heterogeneous, bipartite, temporal
- Node features, edge features, graph-level features

### Graph Machine Learning Tasks
- **Node-level**: node classification (predict properties of individual nodes)
- **Link-level**: link prediction (predict missing edges or edge types)
- **Graph-level**: graph classification/regression (predict a property of the whole graph)

### Message Passing Framework
- The general Message Passing Neural Network (MPNN) paradigm
- Step 1: Message — each node collects messages from neighbors
- Step 2: Aggregate — combine messages (sum, mean, max, LSTM)
- Step 3: Update — update node representation using aggregated messages
- Step 4: Readout — pool node representations for graph-level tasks

### Graph Convolutional Networks (GCN)
- Spectral graph convolution intuition
- Kipf & Welling (2017) formulation: Ã = D̃^{-1/2} Ã D̃^{-1/2}
- Layer-wise propagation rule
- Semi-supervised node classification
- Limitations: transductive (can't generalize to new graphs)

### Graph Attention Networks (GAT)
- Replace fixed aggregation with learned attention coefficients
- Multi-head attention on graphs
- GAT vs. GCN: adaptive neighborhood weighting
- GATv2: more expressive attention mechanism

### GraphSAGE (Inductive Learning)
- Sample a fixed-size neighborhood (scalability)
- Aggregators: mean, LSTM, pooling
- Inductive: can generalize to unseen nodes at test time
- Mini-batch training on large graphs

### Advanced Architectures
- **Graph Transformers**: Graphormer (Microsoft), Graph Transformer (GT)
- **Graph Isomorphism Network (GIN)**: theoretically most expressive MPNN
- **Temporal GNNs**: DyRep, TGAT, TGN for dynamic graphs
- **Heterogeneous GNNs**: HAN, HetGNN for multi-type node/edge graphs
- **Knowledge Graph Embeddings**: TransE, RotatE, ComplEx

### Libraries
- **PyTorch Geometric (PyG)**: the standard GNN library
  - `Data`, `DataLoader`, `HeteroData`
  - Built-in datasets: Cora, CiteSeer, PubMed, OGB benchmarks
  - GNN layers: `GCNConv`, `GATConv`, `SAGEConv`, `GINConv`
  - Global pooling: `global_mean_pool`, `global_add_pool`
- **Deep Graph Library (DGL)**: alternative, supports TF/PyTorch
- **NetworkX**: graph analysis and visualization (not for training)
- **OGB (Open Graph Benchmark)**: standardized datasets and evaluators

### Applications
- **Social Networks**: node classification (bot detection), link prediction (friend recommendation)
- **Drug Discovery**: molecular property prediction (is this molecule toxic?), drug-target interaction
- **Recommendation Systems**: user-item interaction graphs, session-based recommendation
- **Knowledge Graphs**: entity alignment, relation prediction, question answering
- **Traffic Prediction**: road network as graph, speed/flow forecasting
- **Computer Vision**: scene graphs, point cloud processing (PointNet++)

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Project Ideas

1. **Node classification on Cora**: classify academic papers by topic using GCN
2. **Molecule property prediction**: use GIN on MUTAG or QM9 dataset (OGB)
3. **Recommendation with GNNs**: build a bipartite user-item GNN on MovieLens

---

## Further Reading

- [CS224W: Machine Learning with Graphs](https://cs224w.stanford.edu/) — Stanford (free)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- *Graph Representation Learning* — William Hamilton (free PDF)

---

## Next Module

[Module 24: Audio & Speech Processing →](../24-audio-speech-processing/README.md)
