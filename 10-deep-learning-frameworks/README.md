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

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Next Modules

- [Module 11: Computer Vision →](../11-computer-vision/README.md)
- [Module 12: Natural Language Processing →](../12-natural-language-processing/README.md)
