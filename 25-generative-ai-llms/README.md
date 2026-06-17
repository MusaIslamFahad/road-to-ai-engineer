# Module 25: Generative AI & Modern LLMs

**Phase 7 — Generative AI & LLM Applications** | Est. time: 1–2 months (full-time) · 2–4 months (part-time)

---

## 🤖 Companion Resource

> **[ai-integration-hub](https://github.com/MusaIslamFahad/ai-integration-hub)** provides a live API directory, production code templates (OpenAI, Anthropic, LangChain, LlamaIndex, RAG), model comparison tables, and AI engineering roadmaps. Live demo: [aiintegrationhub.vercel.app](https://aiintegrationhub.vercel.app)

---


## Learning Objectives

By the end of this module, you will:
- Engineer prompts effectively using zero-shot, few-shot, and chain-of-thought techniques
- Build end-to-end RAG pipelines with proper evaluation
- Create AI agents with tool use, planning, and memory
- Orchestrate multi-agent systems with CrewAI, AutoGen, and LangGraph
- Deploy and monitor LLM-powered applications in production

---

## Prerequisites

- Module 12: Natural Language Processing (Transformers, fine-tuning)
- Module 10: Deep Learning Frameworks (general proficiency)

---

## Topics Covered

### Prompt Engineering
- Zero-shot: direct instruction, no examples
- Few-shot: provide examples in the prompt
- Chain-of-Thought (CoT): "Let's think step by step"
- ReAct: Reasoning + Acting (interleave thought and tool use)
- Structured output: JSON mode, response schemas, Pydantic validation
- Prompt templates and versioning (LangSmith, PromptLayer)
- System prompts: persona, constraints, output format
- Generative configuration: temperature, top-p, top-k, max_tokens

### Embeddings & Semantic Search
- Embedding models: `text-embedding-3-small`, `embed-english`, `nomic-embed`
- Cosine similarity and nearest-neighbor search
- Dimensionality: trade-offs between quality and cost
- Batch embedding and caching strategies

### Vector Databases
| Database | Best For | Hosting |
|---|---|---|
| **ChromaDB** | Prototyping, local | Local / Cloud |
| **FAISS** | Fast similarity search | Local |
| **Pinecone** | Production, serverless | Cloud |
| **Weaviate** | Hybrid search, GraphQL | Local / Cloud |
| **Qdrant** | High performance, filtering | Local / Cloud |
| **pgvector** | Already using Postgres | Local / Cloud |

### RAG Systems (Retrieval Augmented Generation)
- **Naive RAG**: chunk → embed → retrieve → generate
- **Advanced RAG**:
  - Query expansion and HyDE (hypothetical document embeddings)
  - Hybrid retrieval: dense (semantic) + sparse (BM25)
  - Re-ranking: cross-encoder scoring
  - Contextual compression: extract relevant portions only
  - Parent-child chunking
- **Modular RAG**: routing, adaptive retrieval, self-RAG
- **RAG Evaluation with RAGAS**: faithfulness, answer relevancy, context recall, context precision
- Streaming responses for better UX

### LangChain Framework
- LCEL (LangChain Expression Language): composable chains with `|` pipes
- Memory: `ConversationBufferMemory`, `ConversationSummaryMemory`, entity memory
- Document loaders: PDF, HTML, Word, CSV, GitHub, YouTube transcripts
- Text splitters: `RecursiveCharacterTextSplitter`, `SemanticChunker`
- Retrievers: vector store, multi-query, contextual compression, ensemble
- Output parsers: `StrOutputParser`, `JsonOutputParser`, Pydantic models
- Callbacks: logging, LangSmith tracing, token counting

### LlamaIndex
- Documents, nodes, and indices
- Vector store index, summary index, knowledge graph index
- Query engines vs. chat engines
- Sub-question query engine for complex questions
- Metadata filtering and custom retrievers

### AI Agents
- The agent loop: perceive → think → act → observe → repeat
- Tools: define, describe, validate inputs/outputs
- ReAct pattern implementation from scratch
- Function calling / tool use with OpenAI, Anthropic, Gemini APIs
- Memory types: in-context, external (vector DB), episodic
- Planning: task decomposition, plan-and-execute, Tree of Thoughts

### Multi-Agent Systems
- **CrewAI**: agents with roles, tasks, tools, and backstory; sequential and hierarchical processes
- **AutoGen**: agent group chat; AssistantAgent + UserProxyAgent; conversation patterns
- **LangGraph**: stateful graphs; nodes, edges, conditional branching; human-in-the-loop
- Agent coordination patterns: orchestrator-subagent, peer collaboration, hierarchical
- **MCP (Model Context Protocol)**: standard protocol for tool servers; client-server architecture
- **A2A (Agent-to-Agent)**: Google's inter-agent communication standard

### LLM Fine-Tuning
- When to fine-tune vs. prompt vs. RAG
- **LoRA**: inject low-rank matrices into attention layers; `r`, `alpha`, `dropout` params
- **QLoRA**: LoRA + 4-bit NF4 quantization via `bitsandbytes`
- **DPO** (Direct Preference Optimization): align to preferences without RL
- **RLHF** overview: reward model + PPO
- Fine-tuning with `transformers`, `trl`, and `unsloth`
- Evaluation: MMLU, HellaSwag, perplexity, custom benchmarks

### Production GenAI
- Observability: LangSmith, Langfuse, Arize Phoenix
- Guardrails: NeMo Guardrails, Llama Guard, content filtering
- Cost optimization: caching (semantic cache), batching, model routing
- Rate limiting and retry logic
- Streaming implementations: FastAPI + SSE / WebSockets
- Evaluation pipelines: automated testing, regression testing

---


## 1. Prompt Engineering

```python
from openai import OpenAI
import anthropic

oai_client = OpenAI()
ant_client = anthropic.Anthropic()

# ── Zero-shot ──────────────────────────────────────────────────────────────────
def zero_shot(question: str) -> str:
    resp = oai_client.chat.completions.create(
        model='gpt-4o-mini', temperature=0,
        messages=[
            {'role': 'system', 'content': 'You are an expert ML engineer. Be concise.'},
            {'role': 'user',   'content': question}
        ]
    )
    return resp.choices[0].message.content

# ── Few-shot ───────────────────────────────────────────────────────────────────
few_shot_prompt = """Classify the sentiment. Reply with only POSITIVE, NEGATIVE, or NEUTRAL.

Review: "Fantastic quality, arrived early!"  → POSITIVE
Review: "Broke after one week, very unhappy" → NEGATIVE
Review: "It's okay, nothing special."        → NEUTRAL

Review: "{review}" →"""

# ── Chain-of-Thought ───────────────────────────────────────────────────────────
cot_prompt = """Solve step by step.

A ML team has 3 models with val AUC of 0.91, 0.89, 0.94.
The best model takes 3x longer to train. If training takes 2 hours normally,
what's the total training time if they try all three models?

Let's think step by step:"""

# ── Structured output with Pydantic ───────────────────────────────────────────
from pydantic import BaseModel, Field

class SentimentAnalysis(BaseModel):
    sentiment:  str   = Field(description='POSITIVE, NEGATIVE, or NEUTRAL')
    confidence: float = Field(description='Confidence score 0.0-1.0', ge=0, le=1)
    key_phrases: list[str] = Field(description='Key phrases driving the sentiment')
    reasoning:  str   = Field(description='One sentence explanation')

resp = oai_client.beta.chat.completions.parse(
    model='gpt-4o-mini', temperature=0,
    messages=[{'role': 'user', 'content': f'Analyse: "This product changed my life!"'}],
    response_format=SentimentAnalysis
)
result: SentimentAnalysis = resp.choices[0].message.parsed
print(f"{result.sentiment} ({result.confidence:.0%}) — {result.reasoning}")
```

---

## 2. Embeddings & Vector Search

```python
from openai import OpenAI
import numpy as np
import faiss

client = OpenAI()

def embed(texts: list[str], model='text-embedding-3-small') -> np.ndarray:
    resp = client.embeddings.create(input=texts, model=model)
    return np.array([r.embedding for r in resp.data], dtype=np.float32)

# Embed a corpus
documents = [
    "Gradient descent optimises neural network weights iteratively.",
    "BERT is a pre-trained bidirectional transformer for NLP tasks.",
    "Docker containers package code and dependencies together.",
    "k-means clustering groups data into k clusters by centroid distance.",
]
doc_vecs = embed(documents)

# Build FAISS index (cosine similarity via normalisation + inner product)
faiss.normalize_L2(doc_vecs)
index = faiss.IndexFlatIP(doc_vecs.shape[1])
index.add(doc_vecs)

# Query
query = "How do I train a neural network?"
q_vec = embed([query])
faiss.normalize_L2(q_vec)

scores, ids = index.search(q_vec, k=3)
print("Most relevant documents:")
for score, idx in zip(scores[0], ids[0]):
    print(f"  [{score:.4f}] {documents[idx]}")
```

---

## 3. Complete RAG Pipeline (LangChain)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ── 1. Load & chunk documents ─────────────────────────────────────────────────
loader   = PyPDFLoader('document.pdf')
docs     = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks   = splitter.split_documents(docs)
print(f"{len(docs)} pages → {len(chunks)} chunks")

# ── 2. Embed & store ──────────────────────────────────────────────────────────
embeddings  = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory='./db')

# ── 3. Hybrid retrieval ───────────────────────────────────────────────────────
dense_ret = vectorstore.as_retriever(search_kwargs={'k': 10})
bm25_ret  = BM25Retriever.from_documents(chunks, k=10)
retriever = EnsembleRetriever(
    retrievers=[bm25_ret, dense_ret], weights=[0.4, 0.6]
)

# ── 4. RAG chain ──────────────────────────────────────────────────────────────
llm    = ChatOpenAI(model='gpt-4o-mini', temperature=0)
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
""")

def fmt(docs): return '\n\n'.join(d.page_content for d in docs)

rag_chain = (
    RunnableParallel({'context': retriever | fmt, 'question': RunnablePassthrough()})
    | prompt | llm | StrOutputParser()
)

answer = rag_chain.invoke('What is the main contribution of this paper?')
print(answer)

# ── 5. Evaluate with RAGAS ────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

eval_data = Dataset.from_dict({
    'question':     ['What is the main contribution?'],
    'answer':       [rag_chain.invoke('What is the main contribution?')],
    'contexts':     [[c.page_content for c in retriever.invoke('main contribution')]],
    'ground_truth': ['The paper introduces a new attention mechanism...']
})
scores = evaluate(eval_data, metrics=[faithfulness, answer_relevancy, context_recall])
print(scores)
```

---

## 4. AI Agents with Tool Use

```python
from openai import OpenAI
import json, math

client = OpenAI()

# ── Define tools ──────────────────────────────────────────────────────────────
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'calculator',
            'description': 'Evaluate a mathematical expression',
            'parameters': {
                'type': 'object',
                'properties': {
                    'expression': {'type': 'string', 'description': 'Python math expression'}
                },
                'required': ['expression']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Search the web for current information',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query'}
                },
                'required': ['query']
            }
        }
    }
]

def calculator(expression: str) -> str:
    try:   return str(eval(expression, {'__builtins__': {}}, vars(math)))
    except Exception as e: return f"Error: {e}"

def web_search(query: str) -> str:
    return f"[Mock search results for: {query}]"   # replace with real API

TOOL_MAP = {'calculator': calculator, 'web_search': web_search}

def run_agent(user_message: str, max_turns: int = 8) -> str:
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant. Use tools when needed.'},
        {'role': 'user',   'content': user_message}
    ]
    for turn in range(max_turns):
        resp = client.chat.completions.create(
            model='gpt-4o-mini', messages=messages,
            tools=TOOLS, tool_choice='auto'
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content    # done

        messages.append(msg)
        for call in msg.tool_calls:
            fn    = call.function.name
            args  = json.loads(call.function.arguments)
            result = TOOL_MAP[fn](**args)
            messages.append({
                'role': 'tool', 'tool_call_id': call.id, 'content': str(result)
            })

    return "Max turns reached"

answer = run_agent("What is sqrt(2) * pi, rounded to 4 decimal places?")
print(answer)
```

---

## 5. Multi-Agent with CrewAI

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()

# ── Define agents ─────────────────────────────────────────────────────────────
researcher = Agent(
    role      = 'ML Research Analyst',
    goal      = 'Find accurate, current information about ML topics',
    backstory  = 'Expert AI researcher tracking the latest in machine learning.',
    tools     = [search_tool],
    llm       = 'gpt-4o-mini',
    verbose   = True,
    max_iter  = 5
)

writer = Agent(
    role      = 'Technical Writer',
    goal      = 'Write clear, engaging ML content for practitioners',
    backstory  = 'Technical writer specialised in AI/ML topics.',
    llm       = 'gpt-4o-mini',
    verbose   = True
)

# ── Define tasks ──────────────────────────────────────────────────────────────
research_task = Task(
    description      = 'Research the latest developments in {topic} from the past month. '
                       'Include key papers, benchmarks, and company announcements.',
    expected_output  = 'A structured research brief with sources and key findings.',
    agent            = researcher
)
writing_task = Task(
    description      = 'Write an 800-word blog post about {topic} for ML practitioners.',
    expected_output  = 'An 800-word technical blog post in markdown.',
    agent            = writer,
    context          = [research_task]
)

# ── Run crew ──────────────────────────────────────────────────────────────────
crew   = Crew(agents=[researcher, writer], tasks=[research_task, writing_task],
              process=Process.sequential, verbose=True)
result = crew.kickoff(inputs={'topic': 'State Space Models (Mamba) for sequence modelling'})
print(result.raw)
```

---

## 6. LangGraph — Stateful Agent Workflow

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

llm   = ChatOpenAI(model='gpt-4o-mini')
tools = [TavilySearchResults(max_results=3)]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: AgentState):
    return {'messages': [llm_with_tools.invoke(state['messages'])]}

def should_continue(state: AgentState) -> str:
    last = state['messages'][-1]
    return 'tools' if last.tool_calls else END

tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)
workflow.add_node('agent', call_model)
workflow.add_node('tools', tool_node)
workflow.set_entry_point('agent')
workflow.add_conditional_edges('agent', should_continue)
workflow.add_edge('tools', 'agent')

app = workflow.compile()
result = app.invoke({'messages': [HumanMessage('What are the top LLMs released in 2025?')]})
print(result['messages'][-1].content)
```

---

## 7. LLM Fine-Tuning with LoRA (SFT)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch

# 4-bit quantisation config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
)

model_name = 'meta-llama/Llama-3.2-1B'   # small enough for most GPUs
model      = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map='auto'
)
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64, lora_alpha=16, target_modules='all-linear',
    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Format instruction dataset
def format_example(example):
    return (f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}")

dataset = Dataset.from_dict({
    'instruction': ['Explain gradient descent', 'What is overfitting?'],
    'output':      ['Gradient descent is...', 'Overfitting occurs when...']
})

sft_config = SFTConfig(
    output_dir='./qlora-output', num_train_epochs=3,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=2e-4, fp16=True, optim='paged_adamw_8bit',
    max_seq_length=2048, warmup_ratio=0.05
)

trainer = SFTTrainer(
    model=model, train_dataset=dataset, args=sft_config,
    formatting_func=format_example
)
trainer.train()
model.save_pretrained('./qlora-adapter')
```

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Key Resources

| Resource | Description |
|---|---|
| [Generative AI Comprehensive Guide](../resources/generative_ai_comprehensive_guide.md) | Full overview with code |
| [RAG Comprehensive Guide](../resources/rag_comprehensive_guide.md) | RAG architecture to production |
| [Generative AI Comprehensive Guide](../resources/generative_ai_comprehensive_guide.md) | Chains, agents, memory, RAG |
| [AI Agents Guide](../resources/ai_agents_guide.md) | CrewAI, AutoGen, LangGraph, MCP, A2A |

---

## Project: LLM Chatbot & RAG System

Build a production-ready RAG system:
1. Ingest and chunk documents (PDFs, web pages)
2. Store embeddings in ChromaDB or Pinecone
3. Implement hybrid retrieval with re-ranking
4. Add conversation memory
5. Build FastAPI backend with streaming
6. Evaluate with RAGAS
7. Add observability with LangSmith

---

## Next Phase

**[← Module 12](../12-natural-language-processing/README.md)** |  [Phase 7.5: Essential Skills — Module 19 (SQL) →](../19-sql-database-fundamentals/README.md)

