# LlamaIndex Guide

Complete reference for LlamaIndex — the data framework for building LLM applications over private data.

---

## LlamaIndex vs. LangChain

| Dimension | LlamaIndex | LangChain |
|---|---|---|
| **Core focus** | Indexing and querying data | Composing LLM chains and agents |
| **Primary use** | Document Q&A, knowledge bases | Chatbots, agents, tool use, pipelines |
| **Data handling** | First-class indexing + retrieval | Pluggable retriever components |
| **Query engines** | Rich: vector, summary, knowledge graph | Wrapped retriever + chain |
| **Learning curve** | Medium | Medium |
| **Best for** | Complex RAG, multi-document reasoning | General LLM apps, agents |

**Use LlamaIndex for**: sophisticated document indexing, sub-question decomposition, recursive retrieval, knowledge graph queries.  
**Use LangChain for**: agent loops, tool use, memory management, multi-step pipelines.  
**Use both together**: LlamaIndex for retrieval → LangChain for orchestration.

---

## Installation & Setup

```bash
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai \
            llama-index-vector-stores-chroma llama-index-readers-file
```

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm       = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size    = 1024
Settings.chunk_overlap = 50
```

---

## Part 1: Core Abstractions

### Documents and Nodes

```python
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

# ─── Load documents ───────────────────────────────────────────────────────────
# From directory
documents = SimpleDirectoryReader("./docs/").load_data()

# From PDF
from llama_index.readers.file import PDFReader
documents = PDFReader().load_data("paper.pdf")

# Manual document creation
doc = Document(
    text="LlamaIndex is a data framework for LLM applications.",
    metadata={"source": "intro.txt", "author": "LlamaIndex Team", "year": 2024}
)

# ─── Parse into nodes (chunks) ────────────────────────────────────────────────
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
nodes  = parser.get_nodes_from_documents(documents)
print(f"Documents: {len(documents)} → Nodes: {len(nodes)}")

# Semantic splitting (embedding-based)
semantic_parser = SemanticSplitterNodeParser(
    buffer_size=1,
    embed_model=Settings.embed_model
)
semantic_nodes = semantic_parser.get_nodes_from_documents(documents)
```

---

## Part 2: Indexes

### VectorStoreIndex (Most Common)

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# ─── Build from documents (in-memory) ────────────────────────────────────────
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# ─── Persist to Chroma ────────────────────────────────────────────────────────
chroma_client     = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_docs")
vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context   = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# ─── Load existing index ──────────────────────────────────────────────────────
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context
)

# ─── Add new documents to existing index ─────────────────────────────────────
index.insert(Document(text="New document...", metadata={"source": "new.pdf"}))
```

### SummaryIndex

```python
from llama_index.core import SummaryIndex

# Stores all nodes as a list; retrieves all for summarisation tasks
summary_index = SummaryIndex.from_documents(documents)

# Good for: "Summarise the entire document" queries
query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("What is the overall theme of these documents?")
```

### KeywordTableIndex

```python
from llama_index.core import KeywordTableIndex

# Builds a keyword → nodes mapping; good for exact keyword retrieval
kw_index = KeywordTableIndex.from_documents(documents)
query_engine = kw_index.as_query_engine()
response = query_engine.query("What does the paper say about transformers?")
```

---

## Part 3: Query Engines

```python
# ─── Basic query engine ───────────────────────────────────────────────────────
query_engine = index.as_query_engine(
    similarity_top_k=5,              # Retrieve top 5 nodes
    response_mode="compact",         # "compact", "refine", "tree_summarize", "simple_summarize"
    streaming=True                   # Enable streaming
)

response = query_engine.query("What is the main contribution of this paper?")
print(response.response)

# Source nodes used
for node in response.source_nodes:
    print(f"Score: {node.score:.4f} | Source: {node.metadata.get('source')}")
    print(f"  {node.text[:200]}...\n")

# ─── Streaming ────────────────────────────────────────────────────────────────
streaming_engine = index.as_query_engine(streaming=True)
streaming_resp = streaming_engine.query("Explain the methodology.")
streaming_resp.print_response_stream()

# ─── Response modes ───────────────────────────────────────────────────────────
# "compact"          → fit retrieved text into one LLM call; concise
# "refine"           → iterate through each node, refining answer
# "tree_summarize"   → build a tree of summaries; good for long contexts
# "simple_summarize" → combine all chunks, truncate if needed
```

### Sub-Question Query Engine

Decomposes complex questions into sub-questions answered separately.

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex

# Build separate indexes per document or topic
docs_2023 = VectorStoreIndex.from_documents(load_docs("2023"))
docs_2024 = VectorStoreIndex.from_documents(load_docs("2024"))

tools = [
    QueryEngineTool.from_defaults(
        query_engine=docs_2023.as_query_engine(),
        name="docs_2023",
        description="Research papers from 2023"
    ),
    QueryEngineTool.from_defaults(
        query_engine=docs_2024.as_query_engine(),
        name="docs_2024",
        description="Research papers from 2024"
    ),
]

sub_q_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    use_async=True
)

# Decomposes "How did transformer architectures improve from 2023 to 2024?"
# into separate sub-questions answered over each index
response = sub_q_engine.query(
    "How did transformer architectures improve from 2023 to 2024?"
)
```

### RouterQueryEngine

Routes queries to the most appropriate engine.

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Two engines: one for summaries, one for specific facts
summary_engine = summary_index.as_query_engine(response_mode="tree_summarize")
vector_engine  = vector_index.as_query_engine(similarity_top_k=5)

tools = [
    QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        description="Useful for summarising or getting an overview of the document."
    ),
    QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        description="Useful for answering specific questions about the document."
    ),
]

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools
)

response = router_engine.query("Give me a summary of the key findings.")
# → Routed to summary_engine

response = router_engine.query("What exact method was used in experiment 3?")
# → Routed to vector_engine
```

---

## Part 4: Retrievers

```python
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.retrievers import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor

# ─── Basic retriever ──────────────────────────────────────────────────────────
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
nodes = retriever.retrieve("What is RAG?")

# ─── BM25 (keyword) retriever ─────────────────────────────────────────────────
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=10)

# ─── Hybrid retrieval ─────────────────────────────────────────────────────────
from llama_index.core.retrievers import QueryFusionRetriever

hybrid_retriever = QueryFusionRetriever(
    [retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,            # Generate 1 query variant (set > 1 for multi-query)
    mode="reciprocal_rerank", # Combine with RRF
    use_async=True
)

# ─── Post-processing ──────────────────────────────────────────────────────────
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank

reranker = LLMRerank(choice_batch_size=5, top_n=3)
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7),
        reranker
    ]
)
```

---

## Part 5: Chat Engines (Conversational)

```python
from llama_index.core import VectorStoreIndex

# ─── Simple chat engine (maintains conversation history) ─────────────────────
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",   # "condense_question", "context", "simple"
    verbose=True,
    system_prompt="You are a helpful AI assistant. Answer based only on the provided context."
)

response = chat_engine.chat("What is the main topic of the document?")
print(response.response)

response = chat_engine.chat("Tell me more about that.")   # Uses history
print(response.response)

chat_engine.reset()   # Clear history

# ─── Streaming chat ───────────────────────────────────────────────────────────
streaming_chat = index.as_chat_engine(streaming=True)
response = streaming_chat.stream_chat("What are the key findings?")
for token in response.response_gen:
    print(token, end="", flush=True)
```

---

## Part 6: Agents

```python
from llama_index.core.agent import ReActAgent, FunctionCallingAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool

# ─── Function tools ───────────────────────────────────────────────────────────
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"The weather in {city} is 22°C and sunny."

multiply_tool = FunctionTool.from_defaults(fn=multiply)
weather_tool  = FunctionTool.from_defaults(fn=get_weather)

# ─── Query engine tool ────────────────────────────────────────────────────────
doc_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(),
    name="knowledge_base",
    description="Use this to answer questions about the uploaded documents."
)

# ─── ReAct Agent ─────────────────────────────────────────────────────────────
agent = ReActAgent.from_tools(
    [multiply_tool, weather_tool, doc_tool],
    verbose=True,
    max_iterations=10
)

response = agent.chat("What is 25 multiplied by 47? Also, what's the weather in Tokyo?")
print(response.response)

# ─── Function Calling Agent (for models that support tool use natively) ───────
agent = FunctionCallingAgent.from_tools(
    [multiply_tool, doc_tool],
    verbose=True
)
```

---

## Part 7: Evaluation

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner
)

faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator    = RelevancyEvaluator()

# Single evaluation
query    = "What is the main contribution of the paper?"
response = query_engine.query(query)

faith_result = faithfulness_evaluator.evaluate_response(response=response)
print(f"Faithful: {faith_result.passing} | Score: {faith_result.score}")

relev_result = relevancy_evaluator.evaluate_response(query=query, response=response)
print(f"Relevant: {relev_result.passing} | Score: {relev_result.score}")

# Batch evaluation
runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
    workers=4
)
eval_results = runner.evaluate_queries(
    query_engine,
    queries=["Q1?", "Q2?", "Q3?"]
)
```

---

## Quick Reference

| Task | LlamaIndex Component |
|---|---|
| Load documents | `SimpleDirectoryReader`, `PDFReader`, `WebPageReader` |
| Split into chunks | `SentenceSplitter`, `SemanticSplitterNodeParser` |
| Store embeddings | `VectorStoreIndex` + `ChromaVectorStore` / `PineconeVectorStore` |
| Simple Q&A | `index.as_query_engine()` |
| Multi-doc routing | `RouterQueryEngine` |
| Complex multi-step | `SubQuestionQueryEngine` |
| Conversational | `index.as_chat_engine()` |
| Agent with tools | `ReActAgent.from_tools()` |
| Evaluate RAG | `FaithfulnessEvaluator`, `RelevancyEvaluator` |
| Hybrid retrieval | `QueryFusionRetriever` |
| Reranking | `LLMRerank`, `SentenceTransformerRerank` |
