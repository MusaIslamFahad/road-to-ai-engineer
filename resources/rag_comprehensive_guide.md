# RAG Comprehensive Guide

Complete reference for building production-grade Retrieval-Augmented Generation systems.

---

## What is RAG?

RAG grounds LLM responses in your own data. Instead of relying solely on the model's training knowledge, RAG **retrieves** relevant documents at inference time and includes them in the prompt.

```
User Query
    ↓
[Retriever] → Search vector DB → Top-K relevant chunks
    ↓
[Generator] → LLM(query + chunks) → Grounded Answer
```

**Why RAG over fine-tuning?**

| | RAG | Fine-Tuning |
|---|---|---|
| Knowledge updates | Add docs, no retraining | Retrain on new data |
| Cost | Low (API calls) | High (GPU compute) |
| Explainability | Sources traceable | Black box |
| Hallucination | Lower (grounded) | Model-dependent |
| Best for | Factual Q&A, document search | Style, format, behaviour |

---

## Part 1: Document Processing Pipeline

### Loading Documents

```python
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader,
    UnstructuredWordDocumentLoader, CSVLoader,
    GitLoader, YoutubeAudioLoader
)

# PDF
loader = PyPDFLoader("paper.pdf")
docs   = loader.load()            # List[Document]

# Web page
loader = WebBaseLoader("https://docs.example.com/guide")
docs   = loader.load()

# Multiple files
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs   = loader.load()

print(f"Loaded {len(docs)} documents")
print(f"First doc: {docs[0].page_content[:200]}")
print(f"Metadata:  {docs[0].metadata}")
```

### Chunking Strategies

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SemanticChunker
)
from langchain_openai import OpenAIEmbeddings

# ─── Strategy 1: Recursive Character Splitting (default — use this) ───────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Target chunk size in characters
    chunk_overlap=150,      # Overlap to preserve context across chunks
    separators=["\n\n", "\n", ". ", " ", ""]  # Try these separators in order
)
chunks = splitter.split_documents(docs)
print(f"{len(docs)} docs → {len(chunks)} chunks")

# ─── Strategy 2: Token-based (useful for respecting model context limits) ──────
token_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

# ─── Strategy 3: Semantic Chunking (slowest but semantically coherent) ─────────
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",   # Split where embedding distance spikes
    breakpoint_threshold_amount=95
)
semantic_chunks = semantic_splitter.split_documents(docs)

# ─── Strategy 4: Parent-Child Chunking (retrieve small, use large) ─────────────
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma

child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

store      = InMemoryStore()
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
# Retrieves small chunks for relevance → returns large parent for context
```

---

## Part 2: Embedding & Vector Storage

### Embedding Models

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI (paid — best quality)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")   # 1536-dim
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")   # 3072-dim

# Hugging Face (free, self-hosted)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",   # 1024-dim — strong open-source model
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Cohere (paid — good multilingual)
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

### Vector Databases

```python
# ─── ChromaDB: local, great for development ───────────────────────────────────
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_docs"
)
# Load existing
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ─── FAISS: fast, in-memory or disk-based ─────────────────────────────────────
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ─── Pinecone: production serverless ──────────────────────────────────────────
from langchain_pinecone import PineconeVectorStore
import pinecone

pc = pinecone.Pinecone(api_key="YOUR_KEY")
index = pc.Index("my-index")   # Create at console.pinecone.io first
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
vectorstore.add_documents(chunks)
```

---

## Part 3: Retrieval Strategies

### Dense Retrieval (Semantic)

```python
# Basic similarity search
retriever = vectorstore.as_retriever(
    search_type="similarity",          # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5}
)

# MMR (Maximal Marginal Relevance): balance relevance + diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
)

# With metadata filtering
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": {"source": "paper.pdf", "year": "2024"}}
)
```

### Hybrid Retrieval (Dense + Sparse BM25)

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# BM25 sparse retriever (keyword matching)
bm25_retriever = BM25Retriever.from_documents(chunks, k=10)

# Dense retriever
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Combine with Reciprocal Rank Fusion
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]     # Tune: higher weight on BM25 for keyword-heavy domains
)

results = hybrid_retriever.invoke("What is the capital of France?")
```

### Re-ranking

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Cross-encoder: scores (query, doc) pairs more accurately than bi-encoder
reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
compressor = CrossEncoderReranker(model=reranker, top_n=4)

# Retrieve 20, rerank to top 4
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)

# Cohere Rerank (paid, very strong)
from langchain_cohere import CohereRerank
cohere_reranker = CohereRerank(model="rerank-english-v3.0", top_n=4)
```

### Query Expansion & Transformation

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Generate 3 alternative query phrasings, retrieve for each, deduplicate
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# HyDE: generate a hypothetical answer, embed that instead of the query
from langchain.chains import HypotheticalDocumentEmbedder
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    custom_prompt=None  # Uses default HyDE prompt
)
```

---

## Part 4: RAG Chain Assembly

### Standard RAG Chain (LCEL)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions about documents.
Use ONLY the context below to answer. If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

rag_chain = (
    RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the main contribution of this paper?")
```

### RAG with Sources

```python
from langchain_core.runnables import RunnableParallel

rag_chain_with_sources = RunnableParallel(
    {"docs": retriever, "question": RunnablePassthrough()}
).assign(
    answer=(
        lambda x: {"context": format_docs(x["docs"]), "question": x["question"]}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
)

result = rag_chain_with_sources.invoke("Explain the methodology used.")
print("Answer:", result["answer"])
print("\nSources:")
for doc in result["docs"]:
    print(f"  - {doc.metadata.get('source')} (page {doc.metadata.get('page')})")
```

### Conversational RAG (with memory)

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder

# Prompt to re-phrase follow-up questions given chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and the latest user question, rephrase it as a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Main QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using the context below.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Use with chat history
from langchain_core.messages import HumanMessage, AIMessage
chat_history = []

response1 = rag_chain.invoke({"input": "What is the main topic?", "chat_history": chat_history})
chat_history.extend([HumanMessage("What is the main topic?"), AIMessage(response1["answer"])])

response2 = rag_chain.invoke({"input": "Tell me more about that.", "chat_history": chat_history})
```

---

## Part 5: Evaluation with RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # Is answer supported by context?
    answer_relevancy,      # Does answer address the question?
    context_recall,        # How much of the ground truth is in retrieved context?
    context_precision,     # Is retrieved context relevant to the question?
    answer_correctness     # Is answer factually correct vs ground truth?
)
from datasets import Dataset

# Build evaluation dataset
eval_data = {
    "question":     ["What is RAG?", "Who invented transformers?"],
    "answer":       [rag_chain.invoke(q) for q in ["What is RAG?", "Who invented transformers?"]],
    "contexts":     [[doc.page_content for doc in retriever.invoke(q)]
                     for q in ["What is RAG?", "Who invented transformers?"]],
    "ground_truth": ["RAG stands for Retrieval Augmented Generation...",
                     "Transformers were introduced by Vaswani et al. in 2017..."]
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
print(results)
# {'faithfulness': 0.89, 'answer_relevancy': 0.93, 'context_recall': 0.87, 'context_precision': 0.91}
```

### Interpreting RAGAS Scores

| Metric | Measures | Low Score Means |
|---|---|---|
| **Faithfulness** | Answer supported by retrieved context | Hallucination — LLM ignoring context |
| **Answer Relevancy** | Answer addresses the question | Off-topic answers |
| **Context Recall** | Ground truth covered by context | Poor retrieval — missing relevant docs |
| **Context Precision** | Retrieved docs are relevant | Noisy retrieval — irrelevant chunks |

---

## Part 6: Production RAG Checklist

### Ingestion Pipeline
- [ ] Document loading handles all source types (PDF, HTML, Word, CSV)
- [ ] Chunking strategy chosen for content type (technical docs vs. narrative)
- [ ] Metadata preserved: source, page, date, section
- [ ] Embeddings stored in vector DB with metadata filters
- [ ] Incremental indexing (don't re-embed unchanged documents)
- [ ] Ingestion pipeline is idempotent (safe to re-run)

### Retrieval
- [ ] Hybrid search (dense + BM25) for better recall
- [ ] Re-ranking (cross-encoder) for better precision
- [ ] k tuned empirically (start with 5, evaluate)
- [ ] MMR enabled to avoid duplicate chunks

### Generation
- [ ] System prompt clearly instructs to cite sources
- [ ] Response includes source references
- [ ] Streaming enabled for better UX
- [ ] Guardrails: refuse to answer if context insufficient

### Evaluation
- [ ] Golden Q&A test set (50–200 questions with ground truth)
- [ ] RAGAS scores tracked per release
- [ ] Regression testing before each deployment
- [ ] Human evaluation on edge cases

### Observability
- [ ] LangSmith or Langfuse tracing enabled
- [ ] Track: retrieval latency, LLM latency, total latency
- [ ] Log retrieved chunks per query for debugging
- [ ] Alert on faithfulness score drops

---

## Common Failure Modes & Fixes

| Failure | Symptom | Fix |
|---|---|---|
| Poor retrieval | Answer doesn't use relevant info | Hybrid search, better chunking, query expansion |
| Hallucination | Answer contradicts retrieved context | Stricter system prompt, faithfulness guardrails |
| Context overflow | LLM ignores far-away context | Contextual compression, smaller chunks, re-ranking |
| Slow retrieval | High latency at scale | Pinecone/Weaviate, ANN instead of exact search, caching |
| Chunk boundary split | Answer spans two chunks | Larger overlap, parent-child chunking, semantic splitting |
| Stale knowledge | Retrieved docs are outdated | Metadata filtering by date, TTL on embeddings |
