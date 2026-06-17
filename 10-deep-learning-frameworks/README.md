# Module 10: Deep Learning Frameworks

**Phase 5 — Deep Learning Fundamentals** | Est. time: 1–1.5 months (full-time) · 2–3 months (part-time)

---

## Learning Objectives

By the end of this module, you will:
- Write production-quality PyTorch: custom datasets, training loops, GPU acceleration
- Build models in Keras / TensorFlow Functional API
- Export models to ONNX and TorchScript for inference
- Track experiments with Weights & Biases
- Apply mixed precision training for ~2× speedup

---

## Prerequisites

- Module 09: Neural Networks Basics

---

## Topics Covered

### PyTorch Core

| Concept | What to Master |
|---|---|
| **Tensors** | Creation, dtypes, device (cpu/cuda/mps), in-place ops |
| **Autograd** | `requires_grad`, `.backward()`, `.grad`, `torch.no_grad()` |
| **`nn.Module`** | Custom layers, `forward()`, parameter registration |
| **Optimizers** | Adam, AdamW, SGD; `zero_grad()`, `step()` |
| **DataLoader** | `Dataset`, `DataLoader`, `collate_fn`, `pin_memory` |
| **Training Loop** | Train/val/test phases; gradient clipping; checkpointing |

### TensorFlow / Keras
- Sequential, Functional API, Model subclassing
- `tf.data.Dataset` for efficient pipelines
- Callbacks: EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

### Model Export Formats
| Format | Purpose |
|---|---|
| `state_dict` | Resume training, fine-tune |
| TorchScript | Deployment without Python runtime |
| ONNX | Cross-framework inference (ONNX Runtime) |
| Hugging Face | Share pre-trained models on Hub |

### GPU & Mixed Precision
- `model.to(device)` — move model to CUDA/MPS
- `torch.cuda.amp`: `autocast()` + `GradScaler` for fp16 training (~2× speed, ~2× memory saving)

### Weights & Biases
- Experiment tracking, hyperparameter sweeps, artifact versioning
- `wandb.init()`, `wandb.log()`, `wandb.finish()`

---

## 1. PyTorch Tensors & Autograd

```python
import torch
import torch.nn as nn
import numpy as np

# ── Tensor creation ────────────────────────────────────────────────────────────
x = torch.tensor([1.0, 2.0, 3.0])
z = torch.zeros(3, 4)
r = torch.randn(2, 3)
i = torch.eye(4)
device = 'cuda' if torch.cuda.is_available() else \
         'mps'  if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
x = x.to(device)

# ── Autograd: track gradients ─────────────────────────────────────────────────
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()     # y = x0² + x1²
y.backward()           # compute dy/dx
print(x.grad)          # tensor([4., 6.])  — dy/dx = 2x

# Disable gradient tracking (inference / evaluation)
with torch.no_grad():
    pred = model(X_batch)   # no gradient graph built → faster + less memory

# Detach a tensor from the graph
value = loss.detach().item()   # Python float, no grad history
```

---

## 2. Custom Dataset & DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create datasets
train_ds = TabularDataset(X_train, y_train)
test_ds  = TabularDataset(X_test,  y_test)

# DataLoaders — batch, shuffle, parallel loading
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True)

# Iterate
for X_batch, y_batch in train_loader:
    print(X_batch.shape, y_batch.shape)   # (64, n_features), (64,)
    break

# Image dataset from folder structure:
# data/
#   train/class_0/*.jpg
#   train/class_1/*.jpg
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_dataset = datasets.ImageFolder('data/train', transform=transform)
img_loader  = DataLoader(img_dataset, batch_size=32, shuffle=True, num_workers=4)
```

---

## 3. Building Models with nn.Module

```python
import torch
import torch.nn as nn

# ── Sequential (simple stacks) ─────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 10)
)

# ── Custom Module ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=50, hidden_dims=[256, 128, 64], out_dim=2).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 4. Complete Training Loop

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler:  # mixed precision
            from torch.cuda.amp import autocast
            with autocast():
                preds = model(X)
                loss  = criterion(preds, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(X)
            loss  = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(X)
        correct    += (preds.argmax(1) == y).sum().item()
        total      += len(X)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y  = X.to(device), y.to(device)
        preds = model(X)
        loss  = criterion(preds, y)
        total_loss += loss.item() * len(X)
        correct    += (preds.argmax(1) == y).sum().item()
        total      += len(X)
    return total_loss / total, correct / total

# Setup
model     = MLP(in_dim=50, hidden_dims=[256, 128], out_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=1e-2,
                       steps_per_epoch=len(train_loader), epochs=30)
scaler    = torch.cuda.amp.GradScaler() if device == 'cuda' else None

best_val_acc, patience, wait = 0, 5, 0

for epoch in range(30):
    tr_loss, tr_acc  = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
    va_loss, va_acc  = evaluate(model, test_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch+1:2d}: "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
          f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), 'best_model.pt')
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

model.load_state_dict(torch.load('best_model.pt', map_location=device))
print(f"Best val accuracy: {best_val_acc:.4f}")
```

---

## 5. TensorFlow / Keras

```python
import tensorflow as tf
from tensorflow import keras

# ── Sequential ─────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(50,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(2, activation='softmax')
])

# ── Functional API (for complex architectures) ────────────────────────────────
inputs    = keras.Input(shape=(50,))
x         = keras.layers.Dense(256, activation='relu')(inputs)
x         = keras.layers.BatchNormalization()(x)
x         = keras.layers.Dropout(0.3)(x)
outputs   = keras.layers.Dense(2, activation='softmax')(x)
model     = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_keras_model.h5', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.TensorBoard(log_dir='./tb_logs')
]

history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_split=0.1, callbacks=callbacks, verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## 6. Model Serialisation & Export

```python
import torch
import numpy as np

# ── PyTorch: state_dict (recommended) ─────────────────────────────────────────
torch.save(model.state_dict(), 'model_weights.pt')
model.load_state_dict(torch.load('model_weights.pt', map_location=device))
model.eval()

# Save full model (includes architecture)
torch.save(model, 'full_model.pt')
loaded = torch.load('full_model.pt', map_location=device)

# ── TorchScript: deploy without Python ────────────────────────────────────────
model.eval()
scripted = torch.jit.script(model)     # for models with control flow
# OR
dummy_in = torch.randn(1, 50)
traced   = torch.jit.trace(model, dummy_in)   # for simple models
traced.save('model_scripted.pt')

# Load and run anywhere (C++, mobile, no Python)
loaded_script = torch.jit.load('model_scripted.pt')
output = loaded_script(dummy_in)

# ── ONNX: cross-platform inference ────────────────────────────────────────────
dummy_input = torch.randn(1, 50).to(device)
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}},
    opset_version=17
)

import onnxruntime as ort
sess   = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
output = sess.run(None, {'input': X_test[:4].astype(np.float32)})[0]
print(f"ONNX output shape: {output.shape}")
```

---

## 7. Weights & Biases Experiment Tracking

```python
import wandb
import torch

# Initialise run
run = wandb.init(
    project='my-classification',
    name='mlp-v3-adamw',
    config={
        'architecture': 'MLP',
        'hidden_dims': [256, 128],
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'epochs': 30,
        'dropout': 0.3,
    }
)
config = run.config

# Log metrics each epoch
for epoch in range(config.epochs):
    tr_loss, tr_acc = train_epoch(...)
    va_loss, va_acc = evaluate(...)
    wandb.log({
        'epoch': epoch,
        'train/loss': tr_loss, 'train/acc': tr_acc,
        'val/loss':   va_loss, 'val/acc':   va_acc,
        'lr': optimizer.param_groups[0]['lr']
    })

# Log model as artifact
artifact = wandb.Artifact('mlp-model', type='model')
artifact.add_file('best_model.pt')
run.log_artifact(artifact)
wandb.finish()

# Hyperparameter sweep
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/acc', 'goal': 'maximize'},
    'parameters': {
        'lr':          {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
        'dropout':     {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
        'hidden_dim':  {'values': [128, 256, 512]},
        'batch_size':  {'values': [32, 64, 128]},
    }
}
sweep_id = wandb.sweep(sweep_config, project='my-classification')
wandb.agent(sweep_id, function=train_with_config, count=30)
```
---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---


**[← Module 09](../09-neural-networks-basics/README.md)** | **[→ Module 11](../11-computer-vision/README.md)**
