# Module 11: Computer Vision

**Phase 6 — Specialized Deep Learning** | Est. time: 2.5–3.5 months (full-time) · 5–7 months (part-time)

---

## Learning Objectives

By the end of this module, you will:
- Build and train CNNs for image classification from scratch and with transfer learning
- Implement object detection with YOLO and understand two-stage detectors
- Apply semantic and instance segmentation
- Train and understand Generative Adversarial Networks and VAEs
- Work with Diffusion Models including Stable Diffusion
- Understand Vision Transformers (ViT)

---

## Prerequisites

- Module 10: Deep Learning Frameworks (PyTorch proficient)

---

## Topics Covered

### Image Fundamentals
- RGB channels, pixel values, normalization strategies
- Data augmentation: flipping, rotation, crop, color jitter, CutMix, MixUp
- Albumentations library for fast augmentation pipelines
- ImageNet statistics for normalization

### Convolutional Neural Networks
- Convolution operation: kernel, stride, padding, dilation
- Receptive field calculation
- Pooling: MaxPool, AvgPool, Global Average Pooling
- CNN architecture families:
  - **Classic**: LeNet-5, AlexNet, VGG-16/19
  - **Residual**: ResNet-18/34/50/101, ResNeXt
  - **Efficient**: EfficientNet-B0 to B7, MobileNet, ShuffleNet
  - **Modern**: ConvNeXt, RegNet

### Transfer Learning
- Feature extraction: freeze backbone, train head
- Fine-tuning: unfreeze later layers gradually
- When to use which strategy
- `timm` library: 300+ pretrained models in one line

### Object Detection
- **Evaluation**: IoU, mAP@50, mAP@50-95, NMS
- **Two-stage**: R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN
  - Region proposal networks (RPN)
  - RoI pooling and RoI align
- **One-stage**: YOLO v1 → v3 → v5 → v8
  - Anchor boxes, grid cells, objectness scores
  - SSD: multi-scale feature maps
  - RetinaNet: focal loss for class imbalance

### Semantic & Instance Segmentation
- **Semantic**: FCN, U-Net (encoder-decoder + skip connections), DeepLab v3+
- **Instance**: Mask R-CNN (adds mask head to Faster R-CNN)
- **Panoptic**: combining semantic + instance

### Generative Models
- **GANs**: generator, discriminator, minimax game, mode collapse
  - DCGAN: convolutional GAN
  - Conditional GAN: class-conditioned generation
  - CycleGAN: unpaired image-to-image translation
  - StyleGAN2: high-quality face generation
- **VAEs**: encoder → latent space → decoder; reparameterization trick; ELBO loss
- **Diffusion Models**:
  - Forward process: gradually add Gaussian noise
  - Reverse process: denoise with U-Net
  - DDPM, DDIM (faster sampling)
  - Stable Diffusion: latent diffusion, text conditioning with CLIP
  - ControlNet, IP-Adapter for controlled generation

### Vision Transformers
- ViT: patch embeddings, position encoding, class token
- DeiT: distillation for data efficiency
- Swin Transformer: hierarchical, shifted windows
- Connections to NLP Transformers (Module 12)

---


## 1. Image Preprocessing & Augmentation

```python
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# ── torchvision transforms ─────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])   # ImageNet stats
])
val_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── albumentations (faster, more augmentations) ────────────────────────────────
albu_train = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),  # Cutout
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

train_ds  = datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
test_ds   = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
train_ldr = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
test_ldr  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
```

---

## 2. Building a CNN from Scratch

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,  32),  nn.MaxPool2d(2),   # 32×32 → 16×16
            ConvBlock(32, 64),  nn.MaxPool2d(2),   # 16×16 → 8×8
            ConvBlock(64, 128), nn.MaxPool2d(2),   # 8×8   → 4×4
            ConvBlock(128, 256),                    # 4×4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # → 256×1×1
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x): return self.classifier(self.features(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SimpleCNN(num_classes=10).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 3. Transfer Learning (ResNet / EfficientNet)

```python
import torch
import torch.nn as nn
import torchvision.models as models
import timm   # pip install timm

# ── Option A: torchvision ResNet ──────────────────────────────────────────────
backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all layers
for param in backbone.parameters():
    param.requires_grad = False

# Replace final layer for our task
n_features = backbone.fc.in_features
backbone.fc = nn.Sequential(
    nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 10)
)
# Only the new head is trainable
model = backbone.to(device)

# ── Option B: timm (300+ pre-trained models) ──────────────────────────────────
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)

# Fine-tune: unfreeze last 2 blocks
for name, param in model.named_parameters():
    param.requires_grad = 'blocks.5' in name or 'blocks.6' in name or \
                          'classifier' in name or 'head' in name

# Differential learning rates
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'head' not in n], 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

# Label smoothing loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 4. Object Detection with YOLO v8

```python
# pip install ultralytics
from ultralytics import YOLO
import cv2
import torch

# ── Load pre-trained model ─────────────────────────────────────────────────────
model = YOLO('yolov8n.pt')    # n=nano, s=small, m=medium, l=large, x=extra-large

# ── Run inference ──────────────────────────────────────────────────────────────
results = model('dog.jpg', conf=0.25, iou=0.45)
for r in results:
    print(f"Detected {len(r.boxes)} objects")
    for box in r.boxes:
        cls   = int(box.cls[0])
        conf  = float(box.conf[0])
        xyxy  = box.xyxy[0].tolist()   # [x1, y1, x2, y2]
        print(f"  {model.names[cls]:15s}  conf={conf:.2f}  box={[round(v) for v in xyxy]}")

# Draw boxes on image
annotated = results[0].plot()
cv2.imwrite('detected.jpg', annotated)

# ── Fine-tune on custom data ───────────────────────────────────────────────────
# Dataset format: YOLO expects images/ and labels/ directories
# labels/*.txt: one line per object: class_idx cx cy w h (all 0-1 normalised)

model = YOLO('yolov8s.pt')   # start from pre-trained
results = model.train(
    data='my_dataset.yaml',   # YAML with train/val paths and class names
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=1e-3,
    device='0' if torch.cuda.is_available() else 'cpu',
    patience=20,
    augment=True,
    degrees=15.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    project='yolo_runs',
    name='my_detector_v1'
)

# Evaluate mAP
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# Export to ONNX for deployment
model.export(format='onnx', dynamic=True, simplify=True)
```

---

## 5. Semantic Segmentation (U-Net)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, base_ch=64):
        super().__init__()
        c = base_ch
        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c,   c*2)
        self.enc3 = DoubleConv(c*2, c*4)
        self.enc4 = DoubleConv(c*4, c*8)
        # Bottleneck
        self.bottleneck = DoubleConv(c*8, c*16)
        # Decoder
        self.up4 = nn.ConvTranspose2d(c*16, c*8,  2, 2)
        self.dec4 = DoubleConv(c*16, c*8)
        self.up3 = nn.ConvTranspose2d(c*8,  c*4,  2, 2)
        self.dec3 = DoubleConv(c*8,  c*4)
        self.up2 = nn.ConvTranspose2d(c*4,  c*2,  2, 2)
        self.dec2 = DoubleConv(c*4,  c*2)
        self.up1 = nn.ConvTranspose2d(c*2,  c,    2, 2)
        self.dec1 = DoubleConv(c*2,  c)
        self.out  = nn.Conv2d(c, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d  = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d  = self.dec3(torch.cat([self.up3(d),  e3], dim=1))
        d  = self.dec2(torch.cat([self.up2(d),  e2], dim=1))
        d  = self.dec1(torch.cat([self.up1(d),  e1], dim=1))
        return self.out(d)

model = UNet(in_channels=3, n_classes=21).to(device)   # 21 for VOC
# Loss for segmentation: CrossEntropyLoss per pixel (ignore_index=255 for VOC)
criterion = nn.CrossEntropyLoss(ignore_index=255)
```

---

## 6. GAN — Generating Images

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            self._block(latent_dim, features*8, 4, 1, 0),   # 4×4
            self._block(features*8, features*4, 4, 2, 1),   # 8×8
            self._block(features*4, features*2, 4, 2, 1),   # 16×16
            self._block(features*2, features,   4, 2, 1),   # 32×32
            nn.ConvTranspose2d(features, img_channels, 4, 2, 1),  # 64×64
            nn.Tanh()
        )
    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, features, 4, 2, 1), nn.LeakyReLU(0.2),
            self._block(features,   features*2, 4, 2, 1),
            self._block(features*2, features*4, 4, 2, 1),
            self._block(features*4, features*8, 4, 2, 1),
            nn.Conv2d(features*8, 1, 4, 1, 0), nn.Sigmoid()
        )
    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.net(x).view(-1)

LATENT_DIM = 100
G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(50):
    for real_imgs, _ in train_ldr:
        real_imgs = real_imgs.to(device)
        B = real_imgs.size(0)

        # ── Train Discriminator ──────────────────────────────────────────────
        z     = torch.randn(B, LATENT_DIM, 1, 1, device=device)
        fake  = G(z).detach()
        loss_D = criterion(D(real_imgs), torch.ones(B, device=device)) + \
                 criterion(D(fake),      torch.zeros(B, device=device))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # ── Train Generator ──────────────────────────────────────────────────
        z    = torch.randn(B, LATENT_DIM, 1, 1, device=device)
        loss_G = criterion(D(G(z)), torch.ones(B, device=device))
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    print(f"Epoch {epoch+1}: D={loss_D.item():.4f} G={loss_G.item():.4f}")
```

---

## 7. Vision Transformer (ViT) with timm

```python
import timm
import torch

# Load pre-trained ViT
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
model = model.to(device)

# ViT processes images as sequences of patches
# 224×224 image with 16×16 patches → (224/16)² = 196 patches
# Each patch is linearly projected to an embedding dim (768 for ViT-Base)
# + 1 class token = 197 tokens total
# Self-attention across all 197 tokens → global reasoning

# Fine-tune only the head for speed
for name, param in model.named_parameters():
    param.requires_grad = 'head' in name

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)

# OR fully fine-tune with very small lr
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
```

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Projects in This Module

| Project | Skills | Difficulty |
|---|---|---|
| CIFAR-10 Classification | CNNs, Transfer Learning, Data Augmentation | Intermediate |
| Object Detection (YOLO) | YOLO v8, custom dataset, mAP evaluation | Advanced |
| Generative Model (GAN/VAE) | DCGAN, VAE, image generation | Advanced |

---

## Next Module

**[← Module 10](../10-deep-learning-frameworks/README.md)** | **[→ Module 12](../12-natural-language-processing/README.md)**
