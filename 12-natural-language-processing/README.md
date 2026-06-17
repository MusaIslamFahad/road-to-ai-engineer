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


## 1. Text Preprocessing & Classical NLP

```python
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer, WordNetLemmatizer
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stops      = set(stopwords.words('english'))

def preprocess(text: str) -> list[str]:
    text   = text.lower()
    text   = re.sub(r'http\S+',  '', text)   # remove URLs
    text   = re.sub(r'[^a-z\s]', '', text)   # remove punctuation/numbers
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stops and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

text = "The machine learning model was trained on 10,000 examples!"
print(preprocess(text))
# ['machine', 'learning', 'model', 'trained', 'example']

# ── TF-IDF Vectorisation ───────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

tfidf_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),        # unigrams + bigrams
        min_df=2,                  # ignore very rare terms
        max_df=0.95,               # ignore very common terms
        sublinear_tf=True          # log(1+tf) instead of raw tf
    )),
    ('clf', LogisticRegression(C=5, max_iter=500, random_state=42))
])
tfidf_pipe.fit(X_train_texts, y_train)
print(f"TF-IDF LR Accuracy: {tfidf_pipe.score(X_test_texts, y_test):.4f}")
```

---

## 2. Word Embeddings

```python
# ── Word2Vec ───────────────────────────────────────────────────────────────────
from gensim.models import Word2Vec, KeyedVectors
import numpy as np

# Train on your own corpus
sentences = [["machine", "learning", "is", "fun"],
             ["deep", "learning", "uses", "neural", "networks"],
             ["transformers", "revolutionised", "nlp"]]

w2v = Word2Vec(sentences, vector_size=100, window=5,
               min_count=1, workers=4, sg=1)   # sg=1: Skip-gram
print(w2v.wv.most_similar('learning', topn=5))

# Load pre-trained GloVe embeddings
# Download: https://nlp.stanford.edu/projects/glove/
def load_glove(path, dim=100):
    embeddings = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings

glove = load_glove('glove.6B.100d.txt', dim=100)

# Build embedding matrix for a vocabulary
def build_embedding_matrix(vocab, embeddings, dim=100):
    matrix = np.zeros((len(vocab) + 1, dim))
    for i, word in enumerate(vocab):
        if word in embeddings:
            matrix[i + 1] = embeddings[word]
    return matrix

# Sentence embedding: mean pooling of word vectors
def sentence_embedding(text, embeddings, dim=100):
    tokens = text.lower().split()
    vecs   = [embeddings[t] for t in tokens if t in embeddings]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)
```

---

## 3. Self-Attention from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # (B, n_heads, T, d_head)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q = self.split_heads(self.W_q(x))   # (B, h, T, d_head)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn   = self.dropout(F.softmax(scores, dim=-1))
        out    = torch.matmul(attn, V)                    # (B, h, T, d_head)

        # Concat heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm variant (more stable)
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

# Test
B, T, D = 4, 64, 256   # batch=4, seq_len=64, d_model=256
x       = torch.randn(B, T, D)
layer   = TransformerEncoderLayer(d_model=D, n_heads=8)
out     = layer(x)
print(f"Input: {x.shape} → Output: {out.shape}")   # (4, 64, 256)
```

---

## 4. Fine-Tuning BERT for Classification

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
dataset = load_dataset('imdb')   # 25K train, 25K test, binary sentiment

# ── Tokenise ───────────────────────────────────────────────────────────────────
model_name = 'distilbert-base-uncased'
tokenizer  = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True,
                     max_length=512, padding=False)

tokenized      = dataset.map(tokenize, batched=True, remove_columns=['text'])
data_collator  = DataCollatorWithPadding(tokenizer)

# ── Model ──────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2,
    id2label={0: 'NEGATIVE', 1: 'POSITIVE'},
    label2id={'NEGATIVE': 0, 'POSITIVE': 1}
)

# ── Metrics ────────────────────────────────────────────────────────────────────
acc_metric = evaluate.load('accuracy')
f1_metric  = evaluate.load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': acc_metric.compute(predictions=preds, references=labels)['accuracy'],
        'f1':       f1_metric.compute( predictions=preds, references=labels,
                                        average='binary')['f1']
    }

# ── Training arguments ─────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir            = './results/distilbert-imdb',
    num_train_epochs      = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate         = 2e-5,
    weight_decay          = 0.01,
    warmup_ratio          = 0.1,
    evaluation_strategy   = 'epoch',
    save_strategy         = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model  = 'f1',
    fp16                  = True,
    report_to             = 'none',
    logging_steps         = 100,
)

trainer = Trainer(
    model          = model,
    args           = args,
    train_dataset  = tokenized['train'],
    eval_dataset   = tokenized['test'],
    tokenizer      = tokenizer,
    data_collator  = data_collator,
    compute_metrics = compute_metrics,
)
trainer.train()
trainer.save_model('./results/distilbert-imdb/final')
```

---

## 5. LoRA / QLoRA Fine-Tuning (Parameter-Efficient)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = 'bert-base-uncased'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# LoRA config: only inject small matrices into attention layers
lora_config = LoraConfig(
    task_type     = TaskType.SEQ_CLS,
    r             = 16,        # rank — higher = more capacity
    lora_alpha    = 32,        # scaling factor (typically 2×r)
    lora_dropout  = 0.1,
    target_modules = ['query', 'value'],   # BERT attention weight names
    bias          = 'none'
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 592,898 || all params: 109,476,866 || trainable%: 0.5415

# QLoRA: LoRA on a 4-bit quantised model (large LLMs on consumer GPU)
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = 'nf4',
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True
)
large_model = AutoModelForSequenceClassification.from_pretrained(
    'google/flan-t5-large',
    quantization_config = bnb_config,
    device_map          = 'auto'
)
large_model = prepare_model_for_kbit_training(large_model)
large_model = get_peft_model(large_model, lora_config)
```

---

## 6. RAG Pipeline with Hugging Face

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Step 1: Encode knowledge base ─────────────────────────────────────────────
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')

documents = [
    "BERT stands for Bidirectional Encoder Representations from Transformers.",
    "GPT uses a decoder-only transformer trained with causal language modelling.",
    "RAG combines retrieval with generation to ground LLM outputs in external knowledge.",
    "Attention mechanisms allow models to weigh the importance of different input tokens.",
]

doc_embeddings = embedder.encode(documents, normalize_embeddings=True)
print(f"Embeddings shape: {doc_embeddings.shape}")   # (4, 384)

# ── Step 2: Build FAISS index ─────────────────────────────────────────────────
dim   = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)    # Inner Product = cosine similarity (for normalised)
index.add(doc_embeddings.astype(np.float32))
print(f"Index contains {index.ntotal} vectors")

# ── Step 3: Retrieve + Generate ───────────────────────────────────────────────
def rag_answer(query: str, k: int = 2) -> str:
    # Retrieve
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, k)
    retrieved = [documents[i] for i in indices[0]]

    # Build prompt
    context = '\n'.join(f"- {doc}" for doc in retrieved)
    prompt  = f"""Answer the question based only on the context below.

Context:
{context}

Question: {query}
Answer:"""

    # Generate (using a small local model for demo)
    generator = pipeline('text2text-generation', model='google/flan-t5-base')
    response  = generator(prompt, max_new_tokens=100, do_sample=False)[0]
    return response['generated_text']

answer = rag_answer("What is RAG in NLP?")
print(f"Answer: {answer}")
```

---

## 7. Named Entity Recognition (NER)

```python
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    pipeline, DataCollatorForTokenClassification
)
from datasets import load_dataset
import numpy as np

# ── Load CoNLL-2003 NER dataset ────────────────────────────────────────────────
dataset = load_dataset('conll2003')
label_list = dataset['train'].features['ner_tags'].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', ...]

model_name = 'bert-base-cased'   # cased for NER — capitalisation matters!
tokenizer  = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples['tokens'], truncation=True,
                          is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples['ner_tags']):
        word_ids   = tokenized.word_ids(batch_index=i)
        prev_word  = None
        label_ids  = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)         # special tokens → ignore
            elif word_idx != prev_word:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)         # sub-word tokens → ignore
            prev_word = word_idx
        all_labels.append(label_ids)
    tokenized['labels'] = all_labels
    return tokenized

tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels  = len(label_list),
    id2label    = {i: l for i,l in enumerate(label_list)},
    label2id    = {l: i for i,l in enumerate(label_list)}
)

# ── Inference with pipeline ────────────────────────────────────────────────────
ner_pipe = pipeline('ner', model=model, tokenizer=tokenizer,
                    aggregation_strategy='simple')
text    = "Elon Musk founded SpaceX in California in 2002."
results = ner_pipe(text)
for r in results:
    print(f"{r['word']:20s} → {r['entity_group']:5s}  (score: {r['score']:.2f})")
```

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Projects in This Module

| Project | Skills | Difficulty |
|---|---|---|
| Sentiment Analysis on Reviews | Fine-tune BERT, text classification | Advanced |
| LLM Chatbot & RAG System | RAG, LangChain, vector DBs, LLM APIs | Advanced |

---

**[← Module 11](../11-computer-vision/README.md)** | **[→ Module 25](../25-generative-ai-llms/README.md)**
