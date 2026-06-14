# Module 09: Neural Networks Basics

**Phase 5 — Deep Learning Fundamentals** | Est. time: 2 months (full-time) · 4 months (part-time)

---

## Learning Objectives

By the end of this module, you will:
- Understand the mathematics of neural networks: forward pass, loss, backpropagation
- Implement a neural network from scratch in NumPy
- Choose appropriate activation functions, loss functions, and optimizers
- Diagnose and fix training problems (vanishing gradients, overfitting, underfitting)
- Read and understand training curves

---

## Prerequisites

- Modules 00–08 (especially calculus, linear algebra, and classical ML)

---

## Topics Covered

### From Linear to Non-Linear
- Why linear models fail for complex tasks
- Perceptron and the XOR problem
- Multilayer Perceptrons (MLP): hidden layers, universal approximation theorem

### Building Blocks
| Component | Options | When to Use |
|---|---|---|
| **Activation** | Sigmoid, Tanh, ReLU, LeakyReLU, ELU, GELU, Swish | ReLU default for hidden layers; Sigmoid/Softmax for output |
| **Loss** | MSE, MAE, Cross-Entropy, Binary Cross-Entropy, Focal | Task-dependent |
| **Optimizer** | SGD, Momentum, AdaGrad, RMSprop, Adam, AdamW | Adam default; SGD+momentum for research |
| **Regularization** | L1, L2, Dropout, DropConnect, Early Stopping | Combine Dropout + L2 typically |
| **Normalization** | BatchNorm, LayerNorm, GroupNorm, InstanceNorm | BatchNorm for CNNs; LayerNorm for Transformers |

### Backpropagation Deep Dive
- Chain rule in the computation graph
- Manual gradient computation step by step
- Numerical gradient checking
- Computational graphs (static vs. dynamic)

### Optimizers Explained
- Stochastic Gradient Descent (SGD) and mini-batch
- Momentum: accumulate gradient history
- AdaGrad / RMSprop: adaptive learning rates
- Adam: momentum + adaptive rates combined
- AdamW: Adam with decoupled weight decay
- Learning rate scheduling: StepLR, CosineAnnealing, OneCycleLR

### Weight Initialization
- Why it matters (symmetry breaking, gradient flow)
- Xavier/Glorot: for sigmoid/tanh activations
- He/Kaiming: for ReLU activations
- Orthogonal initialization

### Diagnosing Training
- Reading loss/accuracy curves: overfitting, underfitting, instability
- Vanishing gradients: sigmoid stacking, bad initialization
- Exploding gradients: gradient clipping
- Learning rate too high / too low

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Key Exercise

Implement a 2-layer neural network from scratch in NumPy (no PyTorch/TF) that classifies the MNIST digits with >95% accuracy. This forces deep understanding of forward/backward passes.

---

## Next Module

[Module 10: Deep Learning Frameworks →](../10-deep-learning-frameworks/README.md)
