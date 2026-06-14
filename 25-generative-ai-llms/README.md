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

[Phase 7.5: Essential Skills — Module 19 (SQL) →](../19-sql-database-fundamentals/README.md)
