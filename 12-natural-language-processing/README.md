# Module 12: Natural Language Processing

**Phase 6 — Specialized Deep Learning** | Est. time: 2.5–3.5 months (full-time) · 5–7 months (part-time)

---

## Learning Objectives

By the end of this module, you will:
- Build NLP pipelines from tokenization to model predictions
- Implement and understand the full Transformer architecture from attention up
- Fine-tune BERT, RoBERTa, and GPT-style models on custom tasks
- Build RAG (Retrieval Augmented Generation) pipelines
- Work fluently with the Hugging Face ecosystem

---

## Prerequisites

- Module 10: Deep Learning Frameworks (PyTorch)
- Basic understanding of sequence modeling is helpful

---

## Topics Covered

### Classical NLP (Foundation)
- Text preprocessing pipeline: lowercasing, punctuation, stop words
- Tokenization strategies: whitespace, BPE, WordPiece, SentencePiece, Unigram
- Bag-of-Words and TF-IDF: sparse representations
- N-gram models and language modeling basics
- Named Entity Recognition (NER), POS tagging, dependency parsing

### Word Embeddings
- **Word2Vec**: CBOW (predict word from context) and Skip-gram (predict context from word)
- **GloVe**: global co-occurrence matrix factorization
- **FastText**: subword embeddings (handles OOV words)
- Embedding properties: analogies, cosine similarity, semantic spaces

### Sequence Models
- Vanilla RNNs: vanishing gradient problem
- **LSTMs**: cell state, forget gate, input gate, output gate
- **GRUs**: simplified LSTM variant
- Bidirectional RNNs: context from both directions
- Sequence-to-sequence with attention (Bahdanau, Luong)

### The Transformer Architecture
- Self-attention: queries, keys, values; scaled dot-product attention
- Multi-head attention: parallel attention heads
- Positional encoding: sinusoidal and learned
- Feed-forward sublayers; residual connections; layer normalization
- Encoder-only, decoder-only, encoder-decoder variants

### Pre-trained Language Models
| Model | Type | Pre-training | Best For |
|---|---|---|---|
| **BERT** | Encoder | MLM + NSP | Classification, NER, QA |
| **RoBERTa** | Encoder | MLM (longer, larger) | Same as BERT, often better |
| **DeBERTa** | Encoder | Disentangled attention | State-of-the-art classification |
| **GPT-2/3** | Decoder | Causal LM | Text generation |
| **T5** | Enc-Dec | Span corruption | Seq2seq tasks, summarization |
| **BART** | Enc-Dec | Denoising | Summarization, translation |

### Hugging Face Ecosystem
- `AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`
- `Trainer` API: training loop, evaluation, callbacks
- `datasets` library: loading and preprocessing NLP datasets
- `evaluate` library: metrics (BLEU, ROUGE, accuracy, F1)
- Model Hub: finding and using community models

### Fine-Tuning Strategies
- **Full fine-tuning**: update all parameters (expensive but thorough)
- **Head-only**: freeze backbone, train classification head (fast, limited)
- **LoRA**: Low-Rank Adaptation — inject small trainable matrices
- **QLoRA**: LoRA with 4-bit quantization (fine-tune LLMs on a single GPU)
- **Prefix tuning**: prepend trainable tokens
- `PEFT` library for all parameter-efficient methods

### Retrieval Augmented Generation (RAG)
- Why RAG: ground LLM responses in your own data
- Document loading: PDF, HTML, Word, CSV
- Chunking: fixed size, recursive, semantic splitting
- Embedding generation: `sentence-transformers`, OpenAI embeddings
- Vector storage: ChromaDB, FAISS, Weaviate, Pinecone
- Retrieval: dense retrieval, BM25 sparse, hybrid
- Re-ranking: cross-encoder models, Cohere Rerank
- Generation: prompt template + retrieved context + LLM

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Projects in This Module

| Project | Skills | Difficulty |
|---|---|---|
| Sentiment Analysis on Reviews | Fine-tune BERT, text classification | Advanced |
| LLM Chatbot & RAG System | RAG, LangChain, vector DBs, LLM APIs | Advanced |

---

## Next: Generative AI (Phase 7)

[Module 25: Generative AI & LLMs →](../25-generative-ai-llms/README.md)
