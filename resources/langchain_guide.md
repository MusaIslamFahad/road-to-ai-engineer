# LangChain Guide

Complete reference for LangChain — the most widely used framework for building LLM applications.

---

## Installation & Setup

```bash
pip install langchain langchain-openai langchain-anthropic langchain-community \
            langchain-core langchain-text-splitters
```

```python
import os
os.environ["OPENAI_API_KEY"]    = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["LANGCHAIN_API_KEY"] = "ls__..."   # For LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "my-project"
```

---

## Part 1: Chat Models (LLMs)

```python
from langchain_openai    import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000)

# Anthropic
llm = ChatAnthropic(model="claude-opus-4-5", temperature=0)

# Direct invocation
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

response = llm.invoke([
    SystemMessage(content="You are a helpful data scientist."),
    HumanMessage(content="Explain overfitting in 2 sentences.")
])
print(response.content)

# Batch inference
responses = llm.batch([
    [HumanMessage(content="What is RMSE?")],
    [HumanMessage(content="What is MAE?")],
])

# Streaming
for chunk in llm.stream([HumanMessage(content="Tell me about LLMs")]):
    print(chunk.content, end="", flush=True)
```

---

## Part 2: Prompt Templates

```python
from langchain_core.prompts import (
    ChatPromptTemplate, PromptTemplate,
    SystemMessagePromptTemplate, HumanMessagePromptTemplate,
    MessagesPlaceholder, FewShotChatMessagePromptTemplate
)

# ─── Basic chat prompt ────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} consultant. Be concise and precise."),
    ("human",  "Question: {question}\nAnswer in {style} style.")
])
formatted = prompt.format_messages(domain="ML", question="What is dropout?", style="technical")

# ─── With chat history placeholder ───────────────────────────────────────────
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human",  "{input}")
])

# ─── Few-shot prompting ───────────────────────────────────────────────────────
examples = [
    {"input": "happy",  "output": "POSITIVE"},
    {"input": "awful",  "output": "NEGATIVE"},
    {"input": "decent", "output": "NEUTRAL"},
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"), ("ai", "{output}")
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples, example_prompt=example_prompt
)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL."),
    few_shot_prompt,
    ("human", "{text}")
])
```

---

## Part 3: LCEL — LangChain Expression Language

LCEL composes chains with the `|` pipe operator. Every component is a `Runnable`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# ─── Simple chain ─────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("Summarise this in one sentence: {text}")
chain  = prompt | llm | parser
result = chain.invoke({"text": "LangChain is a framework for building LLM apps."})

# ─── Parallel chains ──────────────────────────────────────────────────────────
summary_chain     = ChatPromptTemplate.from_template("Summarise: {doc}") | llm | parser
key_points_chain  = ChatPromptTemplate.from_template("List 3 key points from: {doc}") | llm | parser

parallel = RunnableParallel(
    summary    = summary_chain,
    key_points = key_points_chain
)
result = parallel.invoke({"doc": "Long document text here..."})

# ─── Structured output (Pydantic) ─────────────────────────────────────────────
class SentimentResult(BaseModel):
    sentiment:  str   = Field(description="POSITIVE, NEGATIVE, or NEUTRAL")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning:  str   = Field(description="One-sentence explanation")

struct_chain = (
    ChatPromptTemplate.from_template("Analyse sentiment of: {text}")
    | llm.with_structured_output(SentimentResult)
)
result: SentimentResult = struct_chain.invoke({"text": "This product is amazing!"})
print(result.sentiment, result.confidence)

# ─── Runnable with fallback ────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
primary  = ChatOpenAI(model="gpt-4o")
fallback = ChatAnthropic(model="claude-opus-4-5")
chain_with_fallback = (prompt | primary | parser).with_fallbacks([prompt | fallback | parser])

# ─── Retry on failure ─────────────────────────────────────────────────────────
reliable_chain = chain.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
```

---

## Part 4: Memory

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─── Buffer memory: keep all messages (simple, can get large) ─────────────────
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# ─── Window memory: keep last K exchanges ─────────────────────────────────────
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# ─── Summary memory: summarise old messages to save tokens ────────────────────
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# ─── Use with LCEL (manual memory management) ─────────────────────────────────
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}   # session_id → InMemoryChatMessageHistory

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
chain = prompt | llm | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Chat with persistent memory
config = {"configurable": {"session_id": "user-123"}}
chain_with_history.invoke({"input": "My name is Alice."}, config=config)
chain_with_history.invoke({"input": "What's my name?"}, config=config)
# → "Your name is Alice."
```

---

## Part 5: Document Loaders & Text Splitters

```python
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader,
    DirectoryLoader, UnstructuredWordDocumentLoader,
    CSVLoader, JSONLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, CharacterTextSplitter,
    TokenTextSplitter, MarkdownHeaderTextSplitter
)

# ─── Load ─────────────────────────────────────────────────────────────────────
docs = PyPDFLoader("paper.pdf").load()
docs = WebBaseLoader("https://example.com/docs").load()
docs = DirectoryLoader("./docs/", glob="**/*.txt", loader_cls=TextLoader).load()

# ─── Split ────────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # characters per chunk
    chunk_overlap=150,     # overlap to preserve context
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"{len(docs)} docs → {len(chunks)} chunks")

# Token-aware splitting (respects model context limits)
token_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

# Markdown-aware splitting (respects headers)
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#","H1"), ("##","H2"), ("###","H3")]
)
```

---

## Part 6: Vector Stores & Retrievers

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ─── Build vector store ────────────────────────────────────────────────────────
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
vectorstore = FAISS.from_documents(chunks, embeddings)

# ─── Retrievers ───────────────────────────────────────────────────────────────
# Simple similarity search
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# MMR (diversity)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
)

# Hybrid: dense + BM25
bm25     = BM25Retriever.from_documents(chunks, k=10)
dense    = vectorstore.as_retriever(search_kwargs={"k": 10})
hybrid   = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])

# Reranking
reranker    = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
compressor  = CrossEncoderReranker(model=reranker, top_n=4)
rerank_ret  = ContextualCompressionRetriever(base_compressor=compressor,
                                              base_retriever=hybrid)
```

---

## Part 7: Complete RAG Chain

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source','?')}]\n{doc.page_content}" for doc in docs
    )

rag_prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the context below. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
""")

rag_chain = (
    RunnableParallel({"context": retriever | format_docs,
                      "question": RunnablePassthrough()})
    | rag_prompt | llm | StrOutputParser()
)

answer = rag_chain.invoke("What is the paper's main contribution?")

# With sources
chain_with_sources = RunnableParallel(
    {"docs": retriever, "question": RunnablePassthrough()}
).assign(
    answer=lambda x: (
        {"context": format_docs(x["docs"]), "question": x["question"]}
        | rag_prompt | llm | StrOutputParser()
    )
)
result = chain_with_sources.invoke("Explain the methodology.")
print("Answer:", result["answer"])
print("Sources:", [d.metadata["source"] for d in result["docs"]])
```

---

## Part 8: LangSmith Observability

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = "ls__..."
os.environ["LANGCHAIN_PROJECT"]    = "my-rag-app"

# All LangChain calls are now auto-traced at smith.langchain.com
# View: latency, token counts, retrieved chunks, LLM inputs/outputs

# Manual feedback
from langsmith import Client
client = Client()
client.create_feedback(run_id="...", key="accuracy", score=1, comment="Correct answer")
```

---

## LCEL Runnable Interface Cheatsheet

| Method | What It Does |
|---|---|
| `.invoke(input)` | Single synchronous call |
| `.batch([input1, input2])` | Multiple inputs in parallel |
| `.stream(input)` | Stream output chunks |
| `.ainvoke(input)` | Async single call |
| `.astream(input)` | Async stream |
| `.with_config({"run_name": "x"})` | Add metadata |
| `.with_retry(stop_after_attempt=3)` | Auto-retry on failure |
| `.with_fallbacks([other_chain])` | Fallback on exception |
| `.with_structured_output(Schema)` | Force Pydantic output |
| `.bind(param=val)` | Partially bind parameters |
| `RunnablePassthrough()` | Pass input unchanged |
| `RunnableParallel({})` | Run multiple chains in parallel |
| `RunnableLambda(fn)` | Wrap any function as Runnable |
