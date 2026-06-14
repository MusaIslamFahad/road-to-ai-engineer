# Transformer Fine-Tuning Guide

Complete guide to fine-tuning BERT, GPT, T5, and modern LLMs using Hugging Face Transformers.

---

## When to Fine-Tune vs. Other Approaches

| Approach | When to Use | Cost | Data Needed |
|---|---|---|---|
| **Zero-shot prompting** | Task is well-known; quick prototype | Near-zero | 0 |
| **Few-shot prompting** | Task needs examples; GPT-4-level model | Low | 5–20 |
| **RAG** | Access to large, up-to-date knowledge base | Low | Documents |
| **Full fine-tuning** | Custom behavior, style, format; data available | High | 1K–1M+ |
| **LoRA / QLoRA** | Same as fine-tuning but cheaper; large models | Medium | 100–100K |
| **Head-only tuning** | Feature extraction; small dataset, similar domain | Low | 100–10K |

---

## Part 1: Fine-Tuning BERT for Classification

### Text Classification (Sentiment, Topic, Spam)

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
import evaluate
import numpy as np

# ─── 1. Load Data ────────────────────────────────────────────────────────────
dataset = load_dataset("imdb")   # or load your own CSV:
# import pandas as pd
# df = pd.read_csv("my_data.csv")
# dataset = Dataset.from_pandas(df)
# dataset = dataset.train_test_split(test_size=0.2)

# ─── 2. Tokenize ─────────────────────────────────────────────────────────────
model_name = "distilbert-base-uncased"   # Faster than BERT; similar quality
tokenizer  = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False          # DataCollatorWithPadding handles dynamic padding
    )

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer)

# ─── 3. Model ────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

# ─── 4. Metrics ──────────────────────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1":       f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
    }

# ─── 5. Training Args ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results/distilbert-imdb",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    report_to="wandb",           # or "none"
    fp16=True,                   # Requires GPU
    dataloader_num_workers=4
)

# ─── 6. Train ────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./results/distilbert-imdb/final")

# ─── 7. Inference ────────────────────────────────────────────────────────────
from transformers import pipeline

classifier = pipeline("text-classification",
                       model="./results/distilbert-imdb/final",
                       tokenizer=tokenizer)
print(classifier("This film was absolutely incredible!"))
# [{'label': 'POSITIVE', 'score': 0.9987}]
```

---

## Part 2: Fine-Tuning for NER (Token Classification)

```python
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

# Labels for NER
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",   # Cased for NER — capitalization matters
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True    # tokens is a list of words
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)          # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)          # Ignore sub-word tokens
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

---

## Part 3: Fine-Tuning with LoRA (Parameter-Efficient)

LoRA (Low-Rank Adaptation) freezes the original weights and injects small trainable matrices. Uses ~10% of the parameters of full fine-tuning with similar results.

```python
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

model_name = "meta-llama/Llama-2-7b-hf"   # or any base model
tokenizer  = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                    # Rank — higher = more capacity, more params
    lora_alpha=32,           # Scaling factor (typically 2×r)
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

# Train exactly the same as normal
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
model.save_pretrained("./lora-adapter")

# Merge for inference (optional — creates single model file)
merged = PeftModel.from_pretrained(base_model, "./lora-adapter")
merged = merged.merge_and_unload()
merged.save_pretrained("./merged-model")
```

---

## Part 4: QLoRA — Fine-Tune Large LLMs on 1 GPU

QLoRA = LoRA + 4-bit quantization. Fine-tune a 7B model on a single 24GB GPU (or a 13B on 2×24GB).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ─── 4-bit quantization config ───────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best quality
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True       # Double quantization for extra savings
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False

# ─── Prepare model for training ──────────────────────────────────────────────
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ─── Format data as instruction-following ───────────────────────────────────
def format_instruction(example):
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

# ─── SFTTrainer (Supervised Fine-Tuning) ─────────────────────────────────────
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./qlora-llama2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch = 4×4 = 16
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    optim="paged_adamw_8bit",           # Memory-efficient optimizer
    max_seq_length=2048,
    warmup_ratio=0.05,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=format_instruction,
)
trainer.train()
```

---

## Part 5: Evaluating Fine-Tuned Models

```python
from datasets import load_dataset
from evaluate import load
from transformers import pipeline

# Text classification evaluation
clf = pipeline("text-classification", model="./my-fine-tuned-model")
dataset = load_dataset("your_eval_dataset", split="test")

preds  = [clf(text)[0]["label"] for text in dataset["text"]]
labels = dataset["label"]

accuracy = load("accuracy")
f1       = load("f1")
print(accuracy.compute(predictions=preds, references=labels))
print(f1.compute(predictions=preds, references=labels, average="macro"))

# For generation (BLEU, ROUGE)
rouge = load("rouge")
bleu  = load("bleu")

predictions = ["Generated text 1", "Generated text 2"]
references  = [["Reference text 1"], ["Reference text 2"]]

print(rouge.compute(predictions=predictions, references=[r[0] for r in references]))
print(bleu.compute(predictions=predictions, references=references))
```

---

## Part 6: Pushing to Hugging Face Hub

```python
from huggingface_hub import login
login()   # or set HF_TOKEN env var

# From Trainer
trainer.push_to_hub("your-username/distilbert-imdb-sentiment")

# Manually
model.push_to_hub("your-username/my-fine-tuned-model")
tokenizer.push_to_hub("your-username/my-fine-tuned-model")

# LoRA adapter only (much smaller than full model)
model.save_pretrained("./adapter")
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="./adapter", repo_id="your-username/my-lora-adapter")
```

---

## Memory & Speed Reference

| Model | Method | GPU Memory | Training Time (1 epoch, 10K samples) |
|---|---|---|---|
| DistilBERT (66M) | Full fine-tune | ~4GB | ~5 min |
| BERT-base (110M) | Full fine-tune | ~6GB | ~10 min |
| RoBERTa-large (355M) | Full fine-tune | ~14GB | ~30 min |
| Llama-2-7B | Full fine-tune | ~56GB | Hours |
| Llama-2-7B | LoRA (r=16) | ~20GB | ~1 hour |
| Llama-2-7B | QLoRA (4-bit) | ~10GB | ~1.5 hours |
| Llama-2-13B | QLoRA (4-bit) | ~18GB | ~3 hours |
| Llama-2-70B | QLoRA (4-bit) | ~48GB (2×24GB) | ~8 hours |

*Estimates vary widely by batch size, sequence length, and hardware.*

---

## Common Problems & Solutions

| Problem | Likely Cause | Fix |
|---|---|---|
| CUDA OOM | Batch size too large | Reduce batch + increase `gradient_accumulation_steps` |
| Loss doesn't decrease | LR too low or too high | Try LRs: [1e-5, 3e-5, 5e-5, 2e-4] |
| Model forgets general knowledge | Full fine-tune, small dataset | Use LoRA; add `weight_decay`; reduce epochs |
| NaN loss | LR too high; bad initialization | Reduce LR; add `warmup_steps` |
| Slow training | Small batch, no `fp16` | Enable `fp16=True`; increase batch |
| Poor tokenization | Wrong `max_length` | Profile token lengths; set 95th percentile as max |
