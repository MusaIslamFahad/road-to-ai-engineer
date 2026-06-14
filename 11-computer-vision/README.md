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

[Module 12: Natural Language Processing →](../12-natural-language-processing/README.md)
