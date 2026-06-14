# Math Formulas Reference

Essential mathematics for machine learning — statistics, linear algebra, calculus, and key ML formulas.

---

## Statistics & Probability

### Descriptive Statistics

| Formula | Symbol | Definition |
|---|---|---|
| Mean | μ = (1/n)Σxᵢ | Average of all values |
| Variance | σ² = (1/n)Σ(xᵢ − μ)² | Average squared deviation |
| Sample Variance | s² = (1/(n−1))Σ(xᵢ − x̄)² | Unbiased variance estimate |
| Std Deviation | σ = √σ² | Spread in original units |
| Covariance | Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)] | Joint variability |
| Correlation | ρ(X,Y) = Cov(X,Y)/(σₓσᵧ) | Normalized covariance ∈ [−1,1] |
| Skewness | γ₁ = E[(X−μ)³]/σ³ | Asymmetry of distribution |
| Kurtosis | γ₂ = E[(X−μ)⁴]/σ⁴ − 3 | Tail heaviness (excess) |

### Probability Rules

```
P(A ∪ B) = P(A) + P(B) − P(A ∩ B)           (Addition rule)
P(A ∩ B) = P(A) × P(B|A)                      (Multiplication rule)
P(A|B)   = P(B|A) × P(A) / P(B)               (Bayes' theorem)
P(B)     = Σ P(B|Aᵢ) × P(Aᵢ)                 (Total probability)
```

### Key Distributions

| Distribution | PMF/PDF | Mean | Variance | Use In ML |
|---|---|---|---|---|
| **Bernoulli(p)** | P(X=1)=p, P(X=0)=1−p | p | p(1−p) | Binary outcome |
| **Binomial(n,p)** | C(n,k)pᵏ(1−p)ⁿ⁻ᵏ | np | np(1−p) | Count of successes |
| **Gaussian(μ,σ²)** | (1/σ√2π)exp(−(x−μ)²/2σ²) | μ | σ² | Weights, residuals |
| **Uniform(a,b)** | 1/(b−a) | (a+b)/2 | (b−a)²/12 | Dropout, initialization |
| **Poisson(λ)** | λᵏe⁻λ/k! | λ | λ | Rare events |
| **Exponential(λ)** | λe⁻λˣ | 1/λ | 1/λ² | Time between events |

### Hypothesis Testing

| Test | Null Hypothesis | Use When |
|---|---|---|
| **t-test (1-sample)** | μ = μ₀ | Test if mean equals value; n < 30 |
| **t-test (2-sample)** | μ₁ = μ₂ | Compare two group means |
| **Chi-squared** | Independence | Test association between categories |
| **ANOVA** | μ₁ = μ₂ = ... = μₖ | Compare k group means |
| **KS test** | Same distribution | Compare empirical distributions |

```
p-value: probability of observing test statistic at least as extreme
         under the null hypothesis.
Reject H₀ if p-value < α (typically α = 0.05)

Type I error (α): reject H₀ when it's true (false positive)
Type II error (β): fail to reject H₀ when it's false (false negative)
Power = 1 − β
```

---

## Linear Algebra

### Vector Operations

```
Dot product:    a · b = Σ aᵢbᵢ = |a||b|cos(θ)
Norm (L2):      ||a||₂ = √(Σ aᵢ²)
Norm (L1):      ||a||₁ = Σ |aᵢ|
Cosine sim:     cos(θ) = (a · b) / (||a|| ||b||)
Outer product:  aᵀb = matrix of shape (m × n)
```

### Matrix Operations

```
Transpose:      (AB)ᵀ = BᵀAᵀ
Inverse:        AA⁻¹ = I  (only square, full-rank matrices)
Trace:          tr(A) = Σ Aᵢᵢ = sum of diagonal
Determinant:    det(A) — scalar; det = 0 means singular
Rank:           number of linearly independent rows/columns
```

### Special Matrices

| Matrix | Property | ML Use |
|---|---|---|
| Symmetric | A = Aᵀ | Covariance matrix, kernel matrix |
| Orthogonal | AᵀA = I, A⁻¹ = Aᵀ | Rotation, PCA eigenvectors |
| Positive Definite | xᵀAx > 0 ∀x≠0 | Covariance matrix, kernel |
| Diagonal | Non-zero only on diagonal | Efficient computation |
| Identity | I: Iₙₘ = 1 if n=m, else 0 | Multiplicative identity |

### Eigenvalues & Eigenvectors

```
Av = λv       (v = eigenvector, λ = eigenvalue)

PCA: find eigenvectors of covariance matrix C = (1/n)XᵀX
     Sort by eigenvalue (largest = most variance)
     Project: X_reduced = X × V[:, :k]

SVD: A = UΣVᵀ
     U: left singular vectors (m×m orthogonal)
     Σ: singular values (m×n diagonal)
     V: right singular vectors (n×n orthogonal)
     Relation: singular values = √eigenvalues of AᵀA
```

### Linear System & Least Squares

```
Ax = b               (exact solution if A is invertible)
x = A⁻¹b

Least squares (overdetermined, m > n):
x = (AᵀA)⁻¹Aᵀb     (normal equations)
         ↑
    This is the OLS solution for linear regression!
```

---

## Calculus

### Derivatives

```
d/dx [xⁿ]      = nxⁿ⁻¹                (power rule)
d/dx [eˣ]      = eˣ
d/dx [ln x]    = 1/x
d/dx [sin x]   = cos x
d/dx [f(g(x))] = f'(g(x)) · g'(x)     (chain rule — the core of backprop)
d/dx [f·g]     = f'g + fg'             (product rule)
```

### Activation Function Derivatives

| Activation | f(x) | f'(x) |
|---|---|---|
| Sigmoid | 1/(1+e⁻ˣ) | σ(x)(1−σ(x)) |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | 1−tanh²(x) |
| ReLU | max(0, x) | 1 if x>0, else 0 |
| LeakyReLU | max(αx, x) | 1 if x>0, else α |
| GELU | x·Φ(x) | Φ(x) + xφ(x) |

### Gradient Descent

```
θ ← θ − η · ∇_θ L(θ)         (parameter update rule)

where:
  η = learning rate
  ∇_θ L = gradient of loss w.r.t. parameters
  L = loss function

Stochastic: use one sample per step
Mini-batch: use B samples per step (B = 32, 64, 128...)
Batch:      use all N samples per step
```

### Partial Derivatives & Jacobian

```
For f: Rⁿ → R:    ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

For f: Rⁿ → Rᵐ:  J = [∂fᵢ/∂xⱼ]  (Jacobian matrix, m×n)

Chain rule (vector form): ∂L/∂x = Jᵀ · ∂L/∂y
                                (this IS backpropagation)
```

---

## ML Loss Functions

### Regression

```
MSE (Mean Squared Error):    L = (1/n)Σ(yᵢ − ŷᵢ)²
MAE (Mean Absolute Error):   L = (1/n)Σ|yᵢ − ŷᵢ|
Huber Loss:                  L = 0.5(y−ŷ)²           if |y−ŷ| ≤ δ
                                 δ|y−ŷ| − 0.5δ²      otherwise
```

### Classification

```
Binary Cross-Entropy:
  L = −(1/n)Σ[yᵢ log(ŷᵢ) + (1−yᵢ)log(1−ŷᵢ)]

Categorical Cross-Entropy:
  L = −(1/n)Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)

KL Divergence:
  KL(P||Q) = Σ P(x) log(P(x)/Q(x))
  Measures how much P differs from Q; always ≥ 0

Focal Loss (for imbalanced data):
  FL(pₜ) = −αₜ(1−pₜ)ᵞ log(pₜ)
  (γ > 0 down-weights easy examples)
```

### Regularization Terms

```
L1 (Lasso):     λΣ|wᵢ|          — promotes sparsity
L2 (Ridge):     λΣwᵢ²           — shrinks all weights
ElasticNet:     λ₁Σ|wᵢ| + λ₂Σwᵢ²  — combines both

Total loss = task_loss + regularization_term
```

---

## Key ML Evaluation Formulas

```
Confusion Matrix:           Actual Pos  Actual Neg
  Predicted Pos:               TP          FP
  Predicted Neg:               FN          TN

Accuracy   = (TP + TN) / (TP + FP + FN + TN)
Precision  = TP / (TP + FP)          — of predicted positives, % correct
Recall     = TP / (TP + FN)          — of actual positives, % found
F1         = 2 × (P × R) / (P + R)
F-beta     = (1+β²) × P × R / (β²×P + R)   (β=2 → recall-weighted)

Specificity = TN / (TN + FP)
FPR         = FP / (FP + TN)
TPR (=Recall)= TP / (TP + FN)

ROC-AUC: area under (FPR, TPR) curve; 0.5 = random, 1.0 = perfect
PR-AUC:  area under (Recall, Precision) curve; better for imbalanced

MCC = (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
      ranges [−1, +1]; +1 = perfect; 0 = random
```

---

## Information Theory

```
Entropy:        H(X) = −Σ P(x) log₂ P(x)
                       (measures uncertainty; bits)

Cross-Entropy:  H(P,Q) = −Σ P(x) log Q(x)
                          (used as loss in classification)

KL Divergence:  KL(P||Q) = H(P,Q) − H(P)
                           (extra bits needed to code P using Q)

Information Gain (Decision Trees):
  IG(S, A) = H(S) − Σ (|Sᵥ|/|S|) × H(Sᵥ)

Gini Impurity:
  Gini(S) = 1 − Σ pᵢ²
```

---

## Optimization Algorithms

```
SGD:          θ ← θ − η·g
Momentum:     v ← βv + g;   θ ← θ − η·v
AdaGrad:      G ← G + g²;   θ ← θ − η·g/√(G+ε)
RMSprop:      G ← βG + (1−β)g²;  θ ← θ − η·g/√(G+ε)
Adam:         m ← β₁m + (1−β₁)g          (1st moment)
              v ← β₂v + (1−β₂)g²         (2nd moment)
              m̂ = m/(1−β₁ᵗ)              (bias correction)
              v̂ = v/(1−β₂ᵗ)
              θ ← θ − η·m̂/√(v̂+ε)

Defaults: β₁=0.9, β₂=0.999, ε=1e-8, η=1e-3
```

---

## Attention & Transformer Mathematics

```
Self-Attention:
  Q = XWᵠ,  K = XWᴷ,  V = XWᵛ           (linear projections)
  Attention(Q,K,V) = softmax(QKᵀ/√dₖ) · V

  √dₖ scaling prevents softmax saturation for large dk

Multi-Head Attention:
  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
  MHA = Concat(head₁,...,head_h) W^O

Positional Encoding (sinusoidal):
  PE(pos,2i)   = sin(pos/10000^(2i/d_model))
  PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

Transformer Block:
  x' = LayerNorm(x + MultiHeadAttn(x))
  x'' = LayerNorm(x' + FFN(x'))
  FFN(x) = max(0, xW₁ + b₁)W₂ + b₂      (dimension: d_model → 4d → d_model)
```

---

## Reinforcement Learning

```
Return:         Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... = Σ γᵏrₜ₊ₖ
Value function: V(s) = E[Gₜ | Sₜ=s]
Q-function:     Q(s,a) = E[Gₜ | Sₜ=s, Aₜ=a]
Advantage:      A(s,a) = Q(s,a) − V(s)

Bellman Optimality:
  V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]
  Q*(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q*(s',a')

TD Update (Q-Learning):
  Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]

Policy Gradient (REINFORCE):
  ∇J(θ) = E_π[∇logπ(a|s;θ) · Gₜ]
```
