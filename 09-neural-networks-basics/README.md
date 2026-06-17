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

## 1. From Linear to Non-Linear

```python
import numpy as np

# Why we need activation functions:
# Without them, stacking linear layers = one linear layer
# f(g(x)) = W2(W1x + b1) + b2 = (W2W1)x + (W2b1+b2) — still linear!

# Activation functions
def sigmoid(z):    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_d(z):  s = sigmoid(z); return s * (1 - s)          # derivative

def tanh(z):       return np.tanh(z)
def tanh_d(z):     return 1 - np.tanh(z)**2

def relu(z):       return np.maximum(0, z)
def relu_d(z):     return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01): return np.where(z > 0, z, alpha * z)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))   # numerical stability
    return e / e.sum(axis=1, keepdims=True)

# Rule of thumb:
# Hidden layers → ReLU (or LeakyReLU)
# Binary output → sigmoid
# Multi-class output → softmax
# Regression output → linear (no activation)
```

---

## 2. Neural Network From Scratch (NumPy)

```python
import numpy as np

class NeuralNetwork:
    """Two-layer MLP implemented in pure NumPy."""

    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        # He initialisation for ReLU layers
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, X):
        """Forward pass: compute predictions and cache intermediates."""
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1          # (N, hidden)
        self.A1 = np.maximum(0, self.Z1)          # ReLU
        self.Z2 = self.A1 @ self.W2 + self.b2    # (N, output)
        self.A2 = softmax(self.Z2)                # probabilities
        return self.A2

    def compute_loss(self, y_pred, y_true):
        """Cross-entropy loss with label smoothing."""
        N = y_true.shape[0]
        # Add epsilon to avoid log(0)
        log_probs = -np.log(y_pred[range(N), y_true] + 1e-15)
        return log_probs.mean()

    def backward(self, y_true):
        """Backpropagation: compute gradients via chain rule."""
        N = self.X.shape[0]

        # ── Output layer gradient ──────────────────────────────────────────
        dZ2 = self.A2.copy()
        dZ2[range(N), y_true] -= 1           # softmax + cross-entropy combined
        dZ2 /= N

        dW2 = self.A1.T @ dZ2                # chain rule: ∂L/∂W2
        db2 = dZ2.sum(axis=0, keepdims=True)

        # ── Hidden layer gradient ──────────────────────────────────────────
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)           # ReLU derivative
        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)

        # ── Gradient descent update ────────────────────────────────────────
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000, batch_size=64, verbose=True):
        N = X.shape[0]
        history = []
        for epoch in range(epochs):
            # Mini-batch SGD
            idx = np.random.permutation(N)
            for i in range(0, N, batch_size):
                Xb = X[idx[i:i+batch_size]]
                yb = y[idx[i:i+batch_size]]
                self.forward(Xb)
                self.backward(yb)

            # Track loss on full set
            preds = self.forward(X)
            loss  = self.compute_loss(preds, y)
            acc   = (preds.argmax(1) == y).mean()
            history.append({'loss': loss, 'acc': acc})

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:4d}: loss={loss:.4f}  acc={acc:.4f}")
        return history

# Test on XOR problem (cannot be solved by linear model)
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X    = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

net     = NeuralNetwork(input_dim=2, hidden_dim=32, output_dim=2, lr=0.1)
history = net.train(X_tr, y_tr, epochs=500, batch_size=32)

preds   = net.forward(X_te).argmax(1)
print(f"\nTest Accuracy: {(preds == y_te).mean():.4f}")
```

---

## 3. Loss Functions

```python
import numpy as np

# ── Regression ─────────────────────────────────────────────────────────────────
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    return np.where(residual <= delta,
                    0.5 * residual**2,
                    delta * residual - 0.5 * delta**2).mean()

# ── Classification ─────────────────────────────────────────────────────────────
def binary_cross_entropy(y_true, y_prob):
    eps = 1e-15
    return -np.mean(y_true * np.log(y_prob + eps) +
                    (1 - y_true) * np.log(1 - y_prob + eps))

def categorical_cross_entropy(y_true_onehot, y_prob):
    eps = 1e-15
    return -np.mean(np.sum(y_true_onehot * np.log(y_prob + eps), axis=1))

# Choose based on task:
# Regression            → MSE (penalises outliers) or MAE (robust) or Huber
# Binary classification → Binary Cross-Entropy
# Multi-class           → Categorical Cross-Entropy (with Softmax)
# Imbalanced classes    → Focal Loss (down-weights easy negatives)
```

---

## 4. Optimizers

```python
import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, W, dW):
        return W - self.lr * dW

class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.velocity = {}
    def update(self, W, dW, key='W'):
        if key not in self.velocity:
            self.velocity[key] = np.zeros_like(W)
        self.velocity[key] = self.momentum * self.velocity[key] - self.lr * dW
        return W + self.velocity[key]

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, {}

    def update(self, W, dW, key='W'):
        if key not in self.m:
            self.m[key] = np.zeros_like(W)
            self.v[key] = np.zeros_like(W)
            self.t[key] = 0

        self.t[key] += 1
        t = self.t[key]
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * dW
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * dW**2
        m_hat = self.m[key] / (1 - self.beta1**t)   # bias correction
        v_hat = self.v[key] / (1 - self.beta2**t)
        return W - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# When to use which:
# SGD + Momentum  → Large vision models; research; better final generalisation
# Adam / AdamW    → Default choice; transformers; quick convergence
# AdamW           → Adam + decoupled weight decay → always prefer over Adam
```

---

## 5. Regularisation

```python
import numpy as np

# ── Dropout ────────────────────────────────────────────────────────────────────
def dropout_forward(A, drop_rate=0.5, training=True):
    if not training:
        return A, None                     # no dropout at test time
    mask = (np.random.rand(*A.shape) > drop_rate) / (1 - drop_rate)  # inverted
    return A * mask, mask

def dropout_backward(dA, mask):
    return dA * mask                       # same mask applied in backward pass

# ── L2 Weight Decay ───────────────────────────────────────────────────────────
def total_loss_with_l2(base_loss, W1, W2, lambda_=1e-4):
    l2_penalty = lambda_ * (np.sum(W1**2) + np.sum(W2**2))
    return base_loss + l2_penalty

# Gradient update with L2:
# dW += 2 * lambda_ * W    (adds a pull toward zero)

# ── Batch Normalisation (forward pass) ────────────────────────────────────────
def batch_norm_forward(Z, gamma, beta, eps=1e-8, training=True, momentum=0.1,
                        running_mean=None, running_var=None):
    if training:
        mu    = Z.mean(axis=0)
        var   = Z.var(axis=0)
        Z_hat = (Z - mu) / np.sqrt(var + eps)
        if running_mean is not None:
            running_mean = (1-momentum)*running_mean + momentum*mu
            running_var  = (1-momentum)*running_var  + momentum*var
    else:
        Z_hat = (Z - running_mean) / np.sqrt(running_var + eps)
    return gamma * Z_hat + beta, Z_hat, mu if training else running_mean, \
           var if training else running_var

# ── Early Stopping ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = np.inf
        self.wait       = 0
        self.best_weights = None

    def __call__(self, val_loss, model_weights):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss    = val_loss
            self.wait         = 0
            self.best_weights = model_weights  # save best
        else:
            self.wait += 1
        return self.wait >= self.patience     # True → stop training
```

---

## 6. Weight Initialisation

```python
import numpy as np

def xavier_init(fan_in, fan_out):
    """For sigmoid/tanh — scale by sqrt(2/(fan_in+fan_out))."""
    limit = np.sqrt(2 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def he_init(fan_in, fan_out):
    """For ReLU — scale by sqrt(2/fan_in)."""
    std = np.sqrt(2 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def zero_init(fan_in, fan_out):
    """NEVER do this for W — symmetry breaking fails."""
    return np.zeros((fan_in, fan_out))   # all neurons learn the same thing!

# Why initialisation matters:
# Too small → vanishing gradients (signal shrinks each layer)
# Too large → exploding gradients (values blow up)
# Xavier    → keeps variance constant through sigmoid/tanh layers
# He        → keeps variance constant through ReLU layers
```

---

## 7. Diagnosing Training from Loss Curves

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train loss')
    ax1.plot(epochs, val_losses,   'r-', label='Val loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, train_accs, 'b-', label='Train acc')
    ax2.plot(epochs, val_accs,   'r-', label='Val acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)

# Reading loss curves:
# Both curves high + parallel        → Underfitting (model too simple or not trained enough)
# Train low, Val high (large gap)    → Overfitting (reduce model size or add regularisation)
# Both curves converged at good val  → Good fit
# Val loss increases after plateau   → Start of overfitting (EarlyStopping kicks in here)
# Loss oscillates wildly             → Learning rate too high
# Loss barely moves                  → Learning rate too low or vanishing gradients
```

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Key Exercise

Implement a 2-layer neural network from scratch in NumPy (no PyTorch/TF) that classifies the MNIST digits with >95% accuracy. This forces deep understanding of forward/backward passes.

```python
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

mnist    = fetch_openml('mnist_784', version=1, as_frame=False)
X, y     = mnist.data / 255.0, mnist.target.astype(int)
X_tr, X_te = X[:60000], X[60000:]
y_tr, y_te = y[:60000], y[60000:]

net = NeuralNetwork(input_dim=784, hidden_dim=256, output_dim=10, lr=0.1)
net.train(X_tr, y_tr, epochs=20, batch_size=256)

preds = net.forward(X_te).argmax(1)
print(f"MNIST Test Accuracy: {(preds == y_te).mean():.4f}")   # target: > 0.95
```

---

**[← Module 08](../08-unsupervised-learning/README.md)** | **[→ Module 10](../10-deep-learning-frameworks/README.md)**
