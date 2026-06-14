# AI Agents Guide

A practical guide to building AI agents and multi-agent systems: from single-agent ReAct loops to production multi-agent orchestration with CrewAI, AutoGen, and LangGraph.

---

## What is an AI Agent?

An AI agent is an LLM that can **perceive**, **reason**, **act**, and **observe** in a loop — using tools to interact with the external world until it completes a task.

```
User Task
    ↓
[THOUGHT]  "I need to search for current stock prices"
[ACTION]   web_search("AAPL stock price today")
[OBSERVATION] "Apple Inc. (AAPL) — $213.45 (+1.2%)"
[THOUGHT]  "Now I can answer the user's question"
[ANSWER]   "Apple's current stock price is $213.45..."
```

---

## Agent Components

| Component | Description | Examples |
|---|---|---|
| **LLM Brain** | Reasoning and decision-making | GPT-4o, Claude 3.5, Gemini 1.5 |
| **Tools** | Functions the agent can call | web_search, run_python, read_file, send_email |
| **Memory** | Retaining context across steps | Conversation history, vector DB, external store |
| **Planning** | Decomposing tasks into sub-tasks | ReAct, Plan-and-Execute, Tree of Thoughts |
| **Observation** | Processing tool outputs | Parsing JSON, reading results, error handling |

---

## Part 1: Single Agents

### ReAct Pattern (from scratch)

```python
from openai import OpenAI
import json

client = OpenAI()

# Define tools
def web_search(query: str) -> str:
    # Real implementation would use SerpAPI or Tavily
    return f"Search results for: {query}..."

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    }
]

TOOL_MAP = {"web_search": web_search, "calculator": calculator}

def run_agent(user_message: str, max_turns: int = 10):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Use tools when needed."},
        {"role": "user", "content": user_message}
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        # No tool calls — agent is done
        if not msg.tool_calls:
            return msg.content

        # Execute tool calls
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = TOOL_MAP[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Max turns reached"

# Usage
answer = run_agent("What is the square root of the current year?")
```

### LangChain Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [
    TavilySearchResults(max_results=3),
]

# ReAct prompt template
prompt = PromptTemplate.from_template("""
Answer the following question using available tools.

Tools: {tools}
Tool names: {tool_names}

Format:
Thought: reasoning
Action: tool_name
Action Input: input
Observation: result
... (repeat as needed)
Thought: I know the final answer
Final Answer: answer

Question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
result = executor.invoke({"input": "What companies are in the S&P 500 AI sector?"})
```

### Memory Types

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 1. Buffer memory — keep all messages
buffer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 2. Summary memory — summarize old messages to save context window
summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

# 3. Vector DB memory — semantic retrieval of relevant past exchanges
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
vector_memory = VectorStoreRetrieverMemory(retriever=retriever)
```

---

## Part 2: CrewAI — Role-Based Multi-Agent Teams

CrewAI models AI agents as a professional crew: each agent has a **role**, **goal**, and **backstory**. Agents collaborate on **tasks** in a **crew** with a defined **process**.

### Installation
```bash
pip install crewai crewai-tools
```

### Core Example: Research & Writing Crew

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Tools
search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

# Agents
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate, current information on AI topics",
    backstory="""You are a seasoned AI researcher with expertise in tracking
    the latest developments in machine learning and generative AI.
    You produce thorough, well-sourced research briefs.""",
    tools=[search_tool, web_tool],
    llm="gpt-4o",
    verbose=True,
    max_iter=5
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, engaging technical content for developers",
    backstory="""You are a technical writer who specializes in AI/ML topics.
    You transform complex research into accessible, well-structured articles.""",
    llm="gpt-4o-mini",
    verbose=True
)

editor = Agent(
    role="Senior Editor",
    goal="Ensure accuracy, clarity, and quality of all published content",
    backstory="You have 10 years of experience editing technical publications.",
    llm="gpt-4o-mini"
)

# Tasks
research_task = Task(
    description="""Research the latest developments in {topic} from the past 3 months.
    Include key papers, company announcements, and benchmark results.
    Output a structured research brief with sources.""",
    expected_output="A 500-word research brief with 5+ sources and key findings.",
    agent=researcher
)

writing_task = Task(
    description="""Using the research brief, write a 800-word technical blog post
    about {topic} for an audience of ML practitioners.
    Include code examples where relevant.""",
    expected_output="An 800-word blog post in markdown with code examples.",
    agent=writer,
    context=[research_task]   # Uses output of research_task
)

editing_task = Task(
    description="Review and polish the blog post for accuracy, flow, and clarity.",
    expected_output="A final, publication-ready blog post.",
    agent=editor,
    context=[writing_task]
)

# Crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,   # or Process.hierarchical
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Multimodal LLMs in 2025"})
print(result.raw)
```

### CrewAI Process Types

| Process | Description | Use When |
|---|---|---|
| `Process.sequential` | Tasks execute in order; each can use previous output | Linear workflows, most common |
| `Process.hierarchical` | Manager agent delegates to workers | Complex tasks needing dynamic planning |

### Advanced CrewAI: Tools, Memory, Callbacks

```python
from crewai import Agent, Crew
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    memory=True,               # Enable all memory types
    long_term_memory=LongTermMemory(),
    short_term_memory=ShortTermMemory(),
    entity_memory=EntityMemory(),
    planning=True,             # Let manager plan before execution
    output_log_file="crew_log.txt"
)
```

---

## Part 3: AutoGen — Conversational Multi-Agent Systems

AutoGen models agents as conversational participants. Agents chat with each other to solve problems.

### Installation
```bash
pip install pyautogen
```

### Basic Two-Agent Conversation

```python
import autogen

config_list = [{"model": "gpt-4o-mini", "api_key": "YOUR_KEY"}]

llm_config = {"config_list": config_list, "temperature": 0}

# AssistantAgent: the AI that does the work
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant. Write clean Python code when asked."
)

# UserProxyAgent: represents the user; can execute code
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",       # "ALWAYS", "NEVER", "TERMINATE"
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function that fetches weather data from OpenWeatherMap and plots a 7-day forecast."
)
```

### Group Chat (Multiple Agents)

```python
import autogen

planner = autogen.AssistantAgent(name="Planner", llm_config=llm_config,
    system_message="You break complex tasks into sub-tasks and assign them.")

data_engineer = autogen.AssistantAgent(name="DataEngineer", llm_config=llm_config,
    system_message="You write efficient data pipelines and SQL queries.")

ml_engineer = autogen.AssistantAgent(name="MLEngineer", llm_config=llm_config,
    system_message="You build and train ML models.")

user_proxy = autogen.UserProxyAgent(name="UserProxy", human_input_mode="NEVER",
    code_execution_config={"work_dir": "project"})

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, data_engineer, ml_engineer],
    messages=[],
    max_round=20,
    speaker_selection_method="auto"   # LLM decides who speaks next
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(manager,
    message="Build an end-to-end churn prediction pipeline with data cleaning, feature engineering, and XGBoost.")
```

---

## Part 4: LangGraph — Stateful Agent Graphs

LangGraph models agentic workflows as **state machines** — nodes (functions/agents) and edges (transitions). Perfect for complex, branching workflows with human-in-the-loop.

### Installation
```bash
pip install langgraph langchain-openai
```

### Basic ReAct Agent with LangGraph

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# State definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Setup
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearchResults(max_results=2)]
llm_with_tools = llm.bind_tools(tools)

# Nodes
def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Run
result = app.invoke({"messages": [HumanMessage(content="What is the latest news about GPT-5?")]})
print(result["messages"][-1].content)
```

### Human-in-the-Loop with LangGraph

```python
from langgraph.checkpoint.memory import MemorySaver

# Add checkpointer for persistence and interruption
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

thread = {"configurable": {"thread_id": "session-123"}}

# First run — pauses before tool execution
for event in app.stream({"messages": [HumanMessage("Search for AAPL earnings")]}, thread):
    print(event)

# Human reviews, approves, then continues
for event in app.stream(None, thread):   # Resume from checkpoint
    print(event)
```

### Multi-Agent Supervisor with LangGraph

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# Supervisor decides which agent to call
def supervisor(state):
    response = supervisor_llm.invoke(state["messages"])
    return {"next": response.additional_kwargs.get("next_agent", END)}

# Router
def route(state):
    return state.get("next", END)

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("coder", coder_agent)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route, {
    "researcher": "researcher",
    "coder": "coder",
    END: END
})
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("coder", "supervisor")
```

---

## Part 5: Model Context Protocol (MCP)

MCP is Anthropic's open protocol for connecting AI models to external tools, data sources, and services. Think of it as a standard USB-C port for AI tools.

### Architecture

```
Claude / LLM (MCP Client)
        ↕  JSON-RPC over stdio / HTTP+SSE
MCP Server (exposes tools, resources, prompts)
        ↕
External Service (GitHub, Slack, database, file system...)
```

### MCP Server Types
- **stdio**: local servers communicating via stdin/stdout (no network)
- **HTTP + SSE**: remote servers for web-hosted tools
- **WebSocket**: bidirectional real-time communication

### Building a Simple MCP Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

server = Server("my-tool-server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_weather":
        city = arguments["city"]
        # Real implementation would call weather API
        return [types.TextContent(type="text", text=f"Weather in {city}: 22°C, Sunny")]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Connecting MCP to Claude API

```python
import anthropic

client = anthropic.Anthropic()

# MCP server connection
response = client.beta.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=[],
    mcp_servers=[{
        "type": "stdio",
        "command": "python",
        "args": ["my_mcp_server.py"]
    }],
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)
```

---

## Part 6: Agent Design Patterns

### Pattern 1: Orchestrator + Subagents
```
User → Orchestrator (plans and delegates)
            ├── Researcher Agent (web search, papers)
            ├── Coder Agent (writes/runs code)
            └── Writer Agent (formats output)
```

### Pattern 2: Parallelization
```
Task → Splitter → [Agent 1, Agent 2, Agent 3] → Aggregator → Result
```
Useful when sub-tasks are independent (e.g., analyze 5 documents simultaneously).

### Pattern 3: Critic / Evaluator Loop
```
Generator Agent → Draft
Critic Agent → Reviews draft → Feedback
Generator Agent → Revised draft
[Repeat until Critic approves or N iterations]
```

### Pattern 4: Plan-and-Execute
```
Planner → [task1, task2, task3, ...]
Executor → executes each task in order or parallel
Re-Planner → adjusts plan based on unexpected results
```

### Pattern 5: RAG + Agent Hybrid
```
Query → Intent Classifier
    ├── [Factual] → RAG retrieval → LLM answer
    ├── [Action] → Agent with tools → Execute + respond
    └── [Conversational] → LLM directly
```

---

## Evaluation & Monitoring

```python
# Simple agent evaluation framework
import time
from dataclasses import dataclass

@dataclass
class AgentTrace:
    task: str
    steps: list[dict]
    final_answer: str
    duration_seconds: float
    tool_calls: int
    success: bool

def evaluate_agent(agent_fn, test_cases: list[dict]) -> dict:
    results = []
    for case in test_cases:
        start = time.time()
        trace = agent_fn(case["query"])
        duration = time.time() - start

        # Check if answer contains expected keywords
        success = all(kw.lower() in trace.final_answer.lower()
                     for kw in case.get("expected_keywords", []))
        results.append({
            "task": case["query"],
            "success": success,
            "duration": duration,
            "tool_calls": trace.tool_calls
        })

    return {
        "success_rate": sum(r["success"] for r in results) / len(results),
        "avg_duration": sum(r["duration"] for r in results) / len(results),
        "avg_tool_calls": sum(r["tool_calls"] for r in results) / len(results),
    }
```

**Production monitoring tools**: LangSmith, Langfuse, Arize Phoenix, Helicone

---

## Quick Comparison: CrewAI vs. AutoGen vs. LangGraph

| Dimension | CrewAI | AutoGen | LangGraph |
|---|---|---|---|
| **Mental model** | Professional team with roles | Agents chatting together | Stateful computation graph |
| **Best for** | Business workflows, content pipelines | Code generation, problem solving | Complex branching, human-in-loop |
| **Control** | Opinionated, easy to start | Conversational, flexible | Maximum control, most verbose |
| **State management** | Implicit | Chat history | Explicit TypedDict state |
| **Human-in-loop** | Via callbacks | `human_input_mode` | `interrupt_before` / `interrupt_after` |
| **Learning curve** | Low | Medium | High |
| **Production readiness** | Growing | Good | Excellent |
