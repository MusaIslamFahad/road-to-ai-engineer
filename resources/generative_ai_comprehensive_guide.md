# Generative AI & LLMs — Comprehensive Guide

A complete reference for building production-grade Generative AI applications: from prompt engineering to multi-agent systems.

---

## Part 1: LLM Fundamentals

### How LLMs Work (Conceptual)

Large Language Models are autoregressive: they predict the next token given all previous tokens. Trained on massive text corpora using self-supervised learning (no labels needed), they develop emergent capabilities at scale — reasoning, translation, code generation — that weren't explicitly trained.

### Key Parameters

| Parameter | Effect | Typical Range |
|---|---|---|
| **Temperature** | Randomness of sampling. 0 = deterministic (greedy), 1 = original distribution, >1 = chaotic | 0–2 |
| **top_p (nucleus)** | Sample from smallest set of tokens whose cumulative probability ≥ p | 0.1–1.0 |
| **top_k** | Sample from top-k most probable tokens | 1–100 |
| **max_tokens** | Maximum tokens to generate | 1–128K+ |
| **frequency_penalty** | Penalise tokens proportional to how often they've appeared | 0–2 |
| **presence_penalty** | Penalise tokens that have appeared at all | 0–2 |

```python
# OpenAI API
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert data scientist."},
        {"role": "user",   "content": "Explain gradient descent in 3 sentences."}
    ],
    temperature=0.3,
    max_tokens=300,
    top_p=0.9
)
print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")

# Anthropic API
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=300,
    system="You are an expert data scientist.",
    messages=[{"role": "user", "content": "Explain gradient descent in 3 sentences."}]
)
print(response.content[0].text)
```

### Model Comparison (2025)

| Model | Provider | Context Window | Strengths |
|---|---|---|---|
| GPT-4o | OpenAI | 128K | Balanced, multimodal, fast |
| Claude Opus 4.6 | Anthropic | 200K | Long context, reasoning, safety |
| Claude Sonnet 4.6 | Anthropic | 200K | Speed/quality balance |
| Gemini 1.5 Pro | Google | 1M | Massive context, multimodal |
| Llama-3-70B | Meta (OSS) | 8K | Open-source, self-hostable |
| Mistral Large | Mistral | 32K | European, efficient |
| Qwen-2-72B | Alibaba | 128K | Strong coding, multilingual |

---

## Part 2: Prompt Engineering

### Zero-Shot

```python
prompt = """Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL.
Reply with only the label.

Review: The product arrived damaged and customer service was unhelpful."""
# → NEGATIVE
```

### Few-Shot

```python
prompt = """Classify sentiment. Reply with only POSITIVE, NEGATIVE, or NEUTRAL.

Review: "Fantastic quality, arrived early!"  → POSITIVE
Review: "Broke on first use, very unhappy."  → NEGATIVE
Review: "It's okay, nothing special."        → NEUTRAL

Review: "The packaging was gorgeous but the product itself disappointed me." →"""
```

### Chain-of-Thought (CoT)

```python
prompt = """Solve this step by step.

A data scientist has 3 datasets: 1,200 rows, 850 rows, and 2,100 rows.
She uses 80% for training and 20% for testing.
If she trains on datasets 1 and 2, tests on dataset 3, and the model
achieves 92% accuracy, how many test samples are classified correctly?

Let's think step by step:"""
# Forces the model to reason before answering → much more accurate on math/logic
```

### System Prompt Patterns

```python
# Persona + Constraints + Output Format
system_prompt = """You are a senior ML engineer at a tech company.

Your role:
- Provide technically accurate, concise answers
- Include code examples when helpful
- Flag any assumptions you make
- If asked for opinions, give them clearly labelled as such

Output format:
- Use markdown with code blocks
- Keep responses under 500 words unless asked for detail
- Always end with "Confidence: [High/Medium/Low]"

Never:
- Fabricate citations or paper names
- Recommend deprecated APIs
- Suggest approaches that don't scale"""
```

### Structured Output

```python
from pydantic import BaseModel
from openai import OpenAI

class MovieReview(BaseModel):
    sentiment: str        # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    score: float          # 1.0–10.0
    key_themes: list[str]
    one_line_summary: str

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Analyse this review: {review_text}"}],
    response_format=MovieReview
)
result: MovieReview = response.choices[0].message.parsed
print(result.sentiment, result.score, result.key_themes)
```

---

## Part 3: Embeddings & Semantic Search

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(texts: list[str], model="text-embedding-3-small") -> np.ndarray:
    response = client.embeddings.create(input=texts, model=model)
    return np.array([r.embedding for r in response.data])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare sentences
vecs = embed(["I love machine learning", "I enjoy deep learning", "The weather is nice"])
print(cosine_similarity(vecs[0], vecs[1]))   # ~0.95 (similar meaning)
print(cosine_similarity(vecs[0], vecs[2]))   # ~0.55 (unrelated)

# Batch embed documents for RAG
documents = ["Doc 1 text...", "Doc 2 text...", ...]
doc_embeddings = embed(documents)  # Shape: (N, 1536)
```

### Embedding Model Comparison

| Model | Dimensions | Cost | Speed | Quality |
|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | $ | Fast | Good |
| `text-embedding-3-large` | 3072 | $$ | Medium | Great |
| `nomic-embed-text` | 768 | Free (OSS) | Fast | Good |
| `bge-large-en-v1.5` | 1024 | Free (OSS) | Medium | Very Good |

---

## Part 4: RAG Systems

### Basic RAG Pipeline

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─── 1. Load Documents ────────────────────────────────────────────────────────
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# ─── 2. Split into Chunks ─────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)

# ─── 3. Embed & Store ─────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# ─── 4. Build RAG Chain ───────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Answer the question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What are the main findings of the paper?")
```

### Advanced RAG Techniques

#### Hybrid Search (Dense + Sparse)

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Dense retriever
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Sparse retriever (BM25 — keyword matching)
bm25_retriever = BM25Retriever.from_documents(chunks, k=10)

# Combine with Reciprocal Rank Fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]    # Tune these for your domain
)
```

#### Re-ranking

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
compressor = CrossEncoderReranker(model=reranker, top_n=4)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)
```

#### RAG Evaluation with RAGAS

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is gradient descent?", "How does BERT work?"],
    "answer":   ["Gradient descent is...", "BERT is..."],    # Model's answers
    "contexts": [["Context chunk 1...", "Context chunk 2..."], [...]],  # Retrieved chunks
    "ground_truth": ["The correct answer 1", "The correct answer 2"]
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
print(results)
# {'faithfulness': 0.89, 'answer_relevancy': 0.93, 'context_recall': 0.87, 'context_precision': 0.91}
```

---

## Part 5: Production Streaming API

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

@app.get("/stream")
async def stream_response(question: str):
    async def generate():
        async for chunk in llm.astream([HumanMessage(content=question)]):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
```

---

## Part 6: LLM Evaluation & Observability

### LangSmith Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "my-rag-project"

# All LangChain calls are now automatically traced in LangSmith
result = rag_chain.invoke("What is the document about?")
# → Full trace with latency, tokens, retrieved chunks, LLM call visible at smith.langchain.com
```

### Langfuse Integration (Open-Source Alternative)

```python
from langfuse.openai import openai   # Drop-in replacement for openai

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    langfuse_user_id="user-123",
    langfuse_session_id="session-abc"
)
```

### Custom Evaluation Pipeline

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(question: str, answer: str, context: str) -> dict:
    """Use an LLM to evaluate RAG output quality."""
    judge_prompt = f"""Evaluate this RAG response on a scale of 1-5 for each criterion.
Respond in JSON only.

Question: {question}
Context: {context}
Answer: {answer}

Rate:
- faithfulness: Is the answer supported by the context? (1-5)
- completeness: Does the answer fully address the question? (1-5)
- coherence: Is the answer clear and well-written? (1-5)

JSON:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}
    )
    import json
    return json.loads(response.choices[0].message.content)
```

---

## Part 7: Cost Optimization

### Strategies

| Strategy | Savings | Trade-off |
|---|---|---|
| **Model routing** | 70–90% | Slower for hard queries |
| **Semantic caching** | 30–60% | Cache invalidation complexity |
| **Prompt compression** | 20–40% | Slight quality loss |
| **Smaller models** | 60–90% | Lower capability |
| **Batching** | 50% | Higher latency |

### Semantic Cache with Redis

```python
from langchain.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
import langchain

langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95   # Reuse cached response if similarity ≥ 0.95
)
# Subsequent identical/near-identical queries return instantly for free
```

### Model Router

```python
def smart_router(query: str) -> str:
    """Route to cheap model for simple queries, expensive for complex."""
    simple_indicators = [
        len(query.split()) < 10,
        query.lower().startswith(("what is", "define", "how do i")),
        not any(kw in query.lower() for kw in ["analyze", "compare", "explain why", "design"])
    ]
    if sum(simple_indicators) >= 2:
        return "gpt-4o-mini"   # $0.15/1M tokens
    return "gpt-4o"            # $2.50/1M tokens
```

---

## Quick Reference: LLM Stack Selection

| Layer | Recommended Tools |
|---|---|
| **LLM Provider** | OpenAI, Anthropic, Google, Groq (fast), Ollama (local) |
| **Orchestration** | LangChain (LCEL), LlamaIndex, raw API |
| **Embeddings** | `text-embedding-3-small` (paid), `nomic-embed` (free OSS) |
| **Vector DB** | ChromaDB (local dev), Pinecone (prod serverless), pgvector (existing Postgres) |
| **Agents** | LangGraph (stateful), CrewAI (role-based), AutoGen (conversational) |
| **Evaluation** | RAGAS (RAG), LangSmith (tracing), Langfuse (OSS tracing) |
| **Serving** | FastAPI + streaming SSE, Gradio (demos), LiteLLM (multi-provider proxy) |
| **Monitoring** | Langfuse, Arize Phoenix, Helicone |
| **Caching** | Redis (semantic cache via LangChain) |
