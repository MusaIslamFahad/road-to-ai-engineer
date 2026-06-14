# Prerequisites Cheatsheet

Quick reference for Python fundamentals and essential mathematics used throughout the AI Engineer path.

---

## Python Core

### Data Types & Collections

```python
# Strings
s = "hello world"
s.upper(), s.split(), s.replace("hello", "hi"), s.strip()
f"Value: {x:.2f}"                  # f-string formatting
"x" in s                           # membership test

# Lists
lst = [1, 2, 3, 4, 5]
lst.append(6); lst.extend([7,8]); lst.pop(); lst.insert(0, 0)
lst[1:4]; lst[::-1]                # slicing; reverse
sorted(lst, reverse=True)
[x**2 for x in lst if x % 2 == 0] # list comprehension

# Dictionaries
d = {"a": 1, "b": 2}
d.get("c", 0)                      # safe access with default
d.keys(), d.values(), d.items()
{k: v*2 for k, v in d.items()}    # dict comprehension

# Sets
s1, s2 = {1,2,3}, {2,3,4}
s1 | s2; s1 & s2; s1 - s2        # union, intersection, difference

# Tuples (immutable)
t = (1, 2, 3)
a, b, c = t                       # unpacking
```

### Functions

```python
# Basics
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# *args and **kwargs
def func(*args, **kwargs):
    print(args)    # tuple of positional
    print(kwargs)  # dict of keyword

# Lambda
square = lambda x: x**2
sorted(items, key=lambda x: x["price"], reverse=True)

# map, filter, reduce
squares  = list(map(lambda x: x**2, [1,2,3,4]))
evens    = list(filter(lambda x: x%2==0, range(10)))
from functools import reduce
total = reduce(lambda a,b: a+b, [1,2,3,4,5])

# Closures and decorators
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-t:.3f}s")
        return result
    return wrapper

@timer
def train_model(): pass
```

### Object-Oriented Programming

```python
class Animal:
    species_count = 0              # class variable

    def __init__(self, name: str, sound: str):
        self.name  = name          # instance variable
        self.sound = sound
        Animal.species_count += 1

    def speak(self) -> str:        # instance method
        return f"{self.name} says {self.sound}"

    @classmethod
    def get_count(cls) -> int:     # class method
        return cls.species_count

    @staticmethod
    def is_animal(obj) -> bool:    # static method
        return isinstance(obj, Animal)

    def __repr__(self): return f"Animal(name={self.name!r})"
    def __len__(self):  return len(self.name)
    def __eq__(self, other): return self.name == other.name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Woof")
        self.breed = breed

    def speak(self):               # override
        return f"{super().speak()}! I'm a {self.breed}"
```

### Iterators & Generators

```python
# Generator (memory-efficient lazy evaluation)
def count_up(n):
    for i in range(n):
        yield i

# Generator expression
squares = (x**2 for x in range(1000))   # not computed yet

# Custom iterator
class Countdown:
    def __init__(self, n): self.n = n
    def __iter__(self):    return self
    def __next__(self):
        if self.n <= 0: raise StopIteration
        self.n -= 1; return self.n + 1
```

### File I/O & Context Managers

```python
# Text files
with open("data.txt", "r") as f:
    lines = f.readlines()
with open("out.txt", "w") as f:
    f.write("result\n")

# JSON
import json
with open("config.json", "r") as f: config = json.load(f)
with open("out.json", "w") as f:   json.dump(data, f, indent=2)

# Pickle (binary serialisation)
import pickle
with open("model.pkl", "wb") as f: pickle.dump(model, f)
with open("model.pkl", "rb") as f: model = pickle.load(f)

# CSV with pandas
import pandas as pd
df = pd.read_csv("data.csv")
df.to_csv("out.csv", index=False)
```

### Exception Handling

```python
try:
    result = 10 / x
except ZeroDivisionError as e:
    print(f"Cannot divide by zero: {e}")
except (TypeError, ValueError) as e:
    print(f"Type or value error: {e}")
else:
    print("Success!")         # runs if no exception
finally:
    print("Always runs")      # cleanup

# Custom exception
class ModelNotFittedError(Exception):
    """Raised when predict is called before fit."""
    pass
```

---

## Big O Notation

| Operation | Array | Dict | Set | Sorted Array | Heap |
|---|---|---|---|---|---|
| Access | O(1) | O(1) | — | O(1) | O(1) peek |
| Search | O(n) | O(1) | O(1) | O(log n) binary | O(n) |
| Insert | O(n) | O(1) | O(1) | O(n) maintain sort | O(log n) |
| Delete | O(n) | O(1) | O(1) | O(n) | O(log n) |

**Common complexities**: O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ)

---

## Mathematics Quick Reference

### Linear Algebra

```python
import numpy as np

# Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)                      # dot product: 32
np.linalg.norm(a)                 # L2 norm: 3.742
a / np.linalg.norm(a)             # unit vector

# Matrices
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
A @ B                             # matrix multiply
A.T                               # transpose
np.linalg.inv(A)                  # inverse
np.linalg.det(A)                  # determinant
eigenvalues, eigenvectors = np.linalg.eig(A)
U, S, Vt = np.linalg.svd(A)      # SVD decomposition

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Statistics

```python
import numpy as np
from scipy import stats

data = np.array([2,4,4,4,5,5,7,9])

# Central tendency
np.mean(data)        # 5.0
np.median(data)      # 4.5
stats.mode(data)     # mode

# Spread
np.var(data, ddof=1)          # sample variance
np.std(data, ddof=1)          # sample std dev
np.percentile(data, [25,75])  # quartiles
stats.iqr(data)               # interquartile range

# Distributions
from scipy.stats import norm, binom, poisson
norm.pdf(0, loc=0, scale=1)   # standard normal PDF at x=0
norm.cdf(1.96)                # P(Z ≤ 1.96) ≈ 0.975
norm.ppf(0.975)               # inverse CDF → 1.96

# Hypothesis testing
t_stat, p_value = stats.ttest_ind(group_a, group_b)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
```

### Calculus for ML

```
Derivative rules:
  d/dx [xⁿ]       = nxⁿ⁻¹
  d/dx [eˣ]       = eˣ
  d/dx [ln x]     = 1/x
  d/dx [f(g(x))]  = f'(g(x)) · g'(x)    ← chain rule (= backpropagation)

Gradient descent update:
  θ ← θ − η · ∇_θ L(θ)
  where η = learning rate, ∇_θ L = gradient of loss w.r.t. parameters

Softmax:
  σ(z)_i = exp(z_i) / Σ_j exp(z_j)    → probabilities that sum to 1

Sigmoid:
  σ(x) = 1 / (1 + e⁻ˣ)              → range (0, 1)
  σ'(x) = σ(x)(1 − σ(x))            ← its own derivative
```

---

## Environment Setup Checklist

```bash
# Install Anaconda (recommended)
# https://www.anaconda.com/products/individual

# Create environment
conda create -n ai-engineer python=3.10
conda activate ai-engineer
pip install -r requirements.txt

# Or Python venv
python -m venv ai-engineer
source ai-engineer/bin/activate      # Mac/Linux
ai-engineer\Scripts\activate         # Windows
pip install -r requirements.txt

# Jupyter
pip install jupyter notebook
jupyter notebook                     # Opens at localhost:8888

# VS Code extensions for ML
# Python, Pylance, Jupyter, GitLens, Rainbow CSV

# Git config
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

---

## Common Python Gotchas in ML

```python
# 1. Mutable default arguments
def bad(lst=[]):    lst.append(1); return lst   # Shares across calls!
def good(lst=None): lst = lst or [];  lst.append(1); return lst

# 2. Integer division
5 / 2    # 2.5 (float division in Python 3)
5 // 2   # 2   (integer division)

# 3. Copy vs reference
import numpy as np
a = np.array([1,2,3])
b = a           # reference — changes to b affect a!
c = a.copy()    # real copy

# 4. Random seeds for reproducibility
import numpy as np, random, torch
np.random.seed(42); random.seed(42); torch.manual_seed(42)

# 5. Float precision
0.1 + 0.2 == 0.3          # False!
abs(0.1 + 0.2 - 0.3) < 1e-9   # True — use tolerance comparison
```
