# ML Glossary

A comprehensive reference of AI/ML terminology — from fundamentals to cutting-edge concepts.

---

## A

**Activation Function** — A non-linear function applied after a linear transformation in a neural network. Common choices: ReLU, GELU, sigmoid, tanh. Without activation functions, deep networks reduce to a single linear layer.

**Adversarial Examples** — Inputs crafted to fool a model while appearing normal to humans. A tiny, imperceptible perturbation to an image can cause a classifier to confidently misclassify it.

**Agent (AI)** — An LLM that reasons and takes actions using tools (web search, code execution, APIs) in a loop until a task is complete.

**Attention Mechanism** — A component that allows a model to weigh the importance of different parts of the input when producing each output. Self-attention in Transformers computes relevance between every pair of positions.

**AutoML** — Automated Machine Learning. Tools (AutoKeras, H2O AutoML, AutoGluon) that automatically search for optimal model architectures and hyperparameters.

**AUC-ROC** — Area Under the Receiver Operating Characteristic Curve. A threshold-independent metric for binary classifiers; 0.5 = random, 1.0 = perfect.

---

## B

**Backpropagation** — The algorithm for computing gradients of a loss function with respect to model parameters by applying the chain rule backward through the computation graph.

**Bagging** — Bootstrap AGGregating. Training multiple models on random samples with replacement and averaging their predictions (e.g., Random Forest).

**Batch Normalization** — Normalization of layer inputs to zero mean and unit variance during training. Stabilizes training and allows higher learning rates.

**BERT** (Bidirectional Encoder Representations from Transformers) — A pre-trained encoder Transformer by Google. Uses masked language modeling to learn bidirectional context. Fine-tuned for classification, NER, Q&A.

**Bias (statistical)** — The error from incorrect assumptions in the learning algorithm. High bias = underfitting. Distinct from algorithmic/social bias.

**Boosting** — An ensemble method that builds models sequentially, each correcting errors of the previous (AdaBoost, Gradient Boosting, XGBoost).

**BPE** (Byte Pair Encoding) — A tokenization algorithm that merges the most frequent character pairs into tokens. Used by GPT models.

---

## C

**CatBoost** — Gradient boosting library by Yandex. Handles categorical features natively without encoding. Often competitive with XGBoost and LightGBM.

**Chain-of-Thought (CoT)** — A prompting technique where the LLM is asked to "think step by step," improving performance on reasoning tasks.

**Chunking** — Splitting documents into smaller pieces for RAG systems. Strategies: fixed-size, recursive, semantic, parent-child.

**Classification** — Predicting a discrete label (category) for an input. Binary: 2 classes. Multi-class: 3+ classes. Multi-label: multiple classes simultaneously.

**Concept Drift** — A change in the statistical relationship between features and the target variable in production. Requires model retraining.

**Confusion Matrix** — A table showing TP, FP, FN, TN counts for a classifier. Foundation for precision, recall, F1, and other metrics.

**Constitutional AI** — Anthropic's approach to training helpful, harmless, and honest AI by having the model evaluate its own outputs against a set of principles.

**Contrastive Learning** — A self-supervised learning approach that pulls similar samples together and pushes dissimilar ones apart in embedding space (e.g., SimCLR, CLIP).

**Convolution** — A mathematical operation that applies a learnable filter across an input, producing a feature map. Core operation of CNNs.

**Cross-Entropy Loss** — The most common loss for classification. Measures the difference between predicted probability distribution and true label distribution.

**Cross-Validation** — A model evaluation technique that splits data into K folds, trains on K-1, tests on 1, and averages results across all K rounds.

---

## D

**Data Augmentation** — Artificially expanding training data with transformations (flipping, rotation, color jitter for images; synonym replacement for text). Improves generalization.

**Data Drift** — A change in the statistical distribution of model inputs in production vs. training. Detected with KS test, PSI, or distribution comparison tools.

**Data Leakage** — When information from the test set (or from the future in time series) accidentally influences training, leading to overoptimistic evaluation.

**DCGAN** (Deep Convolutional GAN) — A GAN architecture using convolutional layers. More stable training than vanilla GANs for image generation.

**Diffusion Model** — A generative model that learns to reverse a gradual noise-adding process. State-of-the-art for image generation (Stable Diffusion, DALL-E 3).

**DPO** (Direct Preference Optimization) — An alternative to RLHF for aligning LLMs to human preferences. More stable training than PPO-based RLHF.

**Dropout** — A regularization technique that randomly sets a fraction of neurons to zero during training, preventing co-adaptation and overfitting.

**DVC** (Data Version Control) — Git-like version control for large datasets and ML models. Stores data in cloud (S3, GCS) and tracks metadata in Git.

---

## E

**Early Stopping** — Stopping training when validation performance stops improving, preventing overfitting.

**Embedding** — A dense vector representation of discrete data (words, sentences, users, items). Captures semantic meaning in a continuous space.

**Encoder-Decoder** — A neural network architecture where an encoder compresses input into a latent representation and a decoder generates output from it. Used in T5, BART, seq2seq.

**Ensemble Methods** — Combining multiple models to produce better predictions. Types: bagging (Random Forest), boosting (XGBoost), stacking (meta-learner).

**Epoch** — One full pass through the entire training dataset.

**Exploding Gradients** — When gradients become very large during backpropagation, causing unstable training. Fixed with gradient clipping.

---

## F

**Feature Engineering** — Transforming raw data into features that improve model performance. Includes encoding, scaling, binning, interaction terms, and domain-specific transformations.

**Feature Importance** — Measures how much each feature contributes to model predictions. Methods: impurity-based (tree models), permutation importance, SHAP values.

**Fine-Tuning** — Continuing to train a pre-trained model on task-specific data, updating all or some parameters.

**Focal Loss** — A variant of cross-entropy that down-weights easy examples to focus training on hard negatives. Used in RetinaNet for object detection with class imbalance.

**Foundation Model** — A large model trained on broad data and adapted to many tasks (GPT-4, Claude, Gemini, LLAMA).

---

## G

**GAN** (Generative Adversarial Network) — Two-network architecture: a generator that creates fake data and a discriminator that distinguishes real from fake. Trained adversarially.

**GELU** (Gaussian Error Linear Unit) — An activation function used in modern transformers (BERT, GPT). Smoother than ReLU; f(x) = x·Φ(x).

**Gradient Descent** — Optimization algorithm that iteratively updates parameters in the direction of the negative gradient of the loss function.

**Graph Neural Network (GNN)** — A neural network that operates on graph-structured data using message passing between nodes.

**GPT** (Generative Pre-trained Transformer) — OpenAI's series of decoder-only Transformer models. Autoregressive: predicts next token given all previous tokens.

---

## H

**Hallucination** — When an LLM generates plausible-sounding but factually incorrect information. A key challenge in production LLM systems.

**Hugging Face** — The leading open-source ML platform providing pre-trained models, datasets, and training libraries (Transformers, Datasets, PEFT, TRL).

**Hyperparameter** — Parameters set before training that are not learned from data (learning rate, batch size, number of layers). Optimized via grid search, random search, or Bayesian optimization.

**HyDE** (Hypothetical Document Embeddings) — A RAG technique where an LLM generates a hypothetical answer to improve retrieval quality.

---

## I–L

**Imbalanced Data** — When classes have significantly different frequencies. Addressed with SMOTE, class weights, threshold tuning, or cost-sensitive learning.

**Inductive Bias** — Assumptions built into a model architecture that influence what patterns it can learn (e.g., CNNs assume spatial locality).

**IoU** (Intersection over Union) — Metric for object detection: area of overlap / area of union between predicted and ground-truth bounding boxes.

**Knowledge Distillation** — Training a small "student" model to mimic a large "teacher" model's outputs, producing a compact model with similar performance.

**KV Cache** — Key-Value cache in transformer inference. Stores computed K and V tensors for previously processed tokens, avoiding recomputation. Essential for fast autoregressive generation.

**L1/L2 Regularization** — Penalty terms added to the loss function to prevent overfitting. L1 (Lasso) induces sparsity; L2 (Ridge) shrinks all weights toward zero.

**Latent Space** — A compressed, abstract representation of data learned by an encoder. Meaningful directions in latent space correspond to semantic features.

**Layer Normalization** — Normalizes activations within a single sample across feature dimensions. Used in Transformers (vs. Batch Norm in CNNs).

**LightGBM** — Microsoft's gradient boosting library. Leaf-wise tree growth makes it faster and more memory-efficient than XGBoost for large datasets.

**LIME** — Local Interpretable Model-agnostic Explanations. Explains individual predictions by training a local linear approximation around the sample.

**LLM** (Large Language Model) — A neural language model with billions of parameters trained on massive text corpora (GPT-4, Claude, Llama-2, Gemini).

**LoRA** (Low-Rank Adaptation) — Parameter-efficient fine-tuning that injects small trainable low-rank matrices into transformer layers. Trains ~0.1–1% of parameters.

---

## M–O

**mAP** (mean Average Precision) — Standard metric for object detection. Averaged over multiple IoU thresholds (COCO: mAP@[.50:.95]).

**Markov Decision Process (MDP)** — Mathematical framework for sequential decision-making: states, actions, transitions, rewards, discount factor.

**MCP** (Model Context Protocol) — Anthropic's open protocol for connecting LLMs to external tools, data sources, and services via a standard interface.

**MLflow** — Open-source platform for experiment tracking, model registry, and deployment.

**MLOps** — DevOps practices applied to ML: experiment tracking, data versioning, CI/CD pipelines, model deployment, and monitoring.

**Multi-Head Attention** — Running multiple self-attention operations in parallel (each with different projections), then concatenating results. Allows the model to attend to different aspects simultaneously.

**NLP** (Natural Language Processing) — The field of AI concerned with processing and understanding human language.

**NMS** (Non-Maximum Suppression) — Post-processing step in object detection that removes duplicate bounding boxes by keeping the one with highest confidence within an IoU threshold.

**Overfitting** — When a model learns the training data too well, including noise, and fails to generalize to new data. Signs: low training error, high validation error.

---

## P–R

**PEFT** (Parameter-Efficient Fine-Tuning) — Methods to fine-tune large models by updating only a small fraction of parameters. Includes LoRA, prefix tuning, prompt tuning, adapters.

**Perplexity** — Measure of how well a language model predicts a test corpus. Lower = better. PP = exp(cross-entropy loss).

**Pipeline (sklearn)** — Chains data preprocessing and model training steps into a single object, preventing data leakage and simplifying deployment.

**PPO** (Proximal Policy Optimization) — A reinforcement learning algorithm used in RLHF to fine-tune LLMs with human feedback.

**Precision-Recall Curve** — Plots precision vs. recall at all possible thresholds. Better than ROC-AUC for highly imbalanced datasets.

**Prompt Engineering** — The practice of designing and optimizing prompts to elicit desired behavior from LLMs.

**QLoRA** — Quantized LoRA. Combines 4-bit NF4 quantization with LoRA adapters to fine-tune very large models on consumer-grade GPUs.

**RAG** (Retrieval-Augmented Generation) — Technique that retrieves relevant documents from a knowledge base and includes them in the LLM's context before generating a response.

**RAGAS** — Framework for evaluating RAG systems on faithfulness, answer relevancy, context recall, and context precision.

**ReAct** — Reasoning + Acting. A prompting/agent pattern where the LLM alternates between generating thoughts and taking tool actions.

**Regularization** — Techniques to prevent overfitting: L1/L2 penalties, Dropout, early stopping, data augmentation, batch normalization.

**Reinforcement Learning (RL)** — Learning paradigm where an agent learns by interacting with an environment, receiving rewards for desired behavior.

**RLHF** (Reinforcement Learning from Human Feedback) — Fine-tuning LLMs using a reward model trained on human preference data + PPO. Used to align ChatGPT and Claude.

**ResNet** — Deep residual network. Uses skip connections to allow training of very deep networks (50, 101, 152 layers) without vanishing gradients.

---

## S–Z

**Self-Supervised Learning** — Learning representations without human-labeled data; uses the data itself to create supervision (masked language modeling, contrastive learning).

**SHAP** (SHapley Additive exPlanations) — Game-theoretic framework for explaining model predictions. Computes the contribution of each feature to a specific prediction.

**SMOTE** (Synthetic Minority Over-sampling Technique) — Creates synthetic examples of the minority class by interpolating between existing examples.

**Stable Diffusion** — An open-source latent diffusion model for text-to-image generation. Operates in a compressed latent space for efficiency.

**Stacking** — Ensemble method where predictions from multiple base models are used as features for a meta-learner.

**Supervised Learning** — Learning from labeled examples: learn a mapping from inputs X to targets y.

**t-SNE** — t-distributed Stochastic Neighbor Embedding. Non-linear dimensionality reduction technique for visualization. Preserves local structure; not suitable for downstream ML.

**Tokenization** — Converting text into tokens (sub-words) that are mapped to integer IDs for model input. BPE, WordPiece, and SentencePiece are common algorithms.

**Transfer Learning** — Using a model trained on one task as the starting point for a different task. Dramatically reduces data and compute requirements.

**Transformer** — Neural network architecture using multi-head self-attention. Basis of all modern LLMs (BERT, GPT, T5, Claude, Gemini).

**UMAP** (Uniform Manifold Approximation and Projection) — Non-linear dimensionality reduction. Faster than t-SNE, better preserves global structure, usable for downstream ML.

**Underfitting** — When a model is too simple to capture the underlying patterns in the data. Signs: high training error, small gap between train and validation error.

**Unsupervised Learning** — Learning structure from unlabeled data: clustering, dimensionality reduction, anomaly detection.

**Vanishing Gradients** — When gradients become very small during backpropagation through deep networks, preventing earlier layers from learning. Fixed with ReLU, residual connections, and careful initialization.

**VAE** (Variational Autoencoder) — Generative model that encodes input into a probabilistic latent space. Can generate new samples by sampling from the learned distribution.

**Vector Database** — Specialized database optimized for storing and querying high-dimensional embedding vectors (ChromaDB, Pinecone, Weaviate, FAISS).

**Vision Transformer (ViT)** — Transformer architecture applied to images by splitting them into patches and treating each patch as a token.

**Word2Vec** — Neural network model that learns word embeddings by predicting context words (Skip-gram) or predicting a word from context (CBOW).

**XGBoost** (eXtreme Gradient Boosting) — Optimized gradient boosting implementation. Fast, regularized, handles missing values. Dominant in tabular ML competitions.

**Zero-Shot Learning** — Model performs a task without seeing any examples during inference, relying solely on task description in the prompt.
