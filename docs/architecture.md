# RAG Assistant Architecture

A production-grade AI assistant combining Groq LLM, LangGraph orchestration, Qdrant vector database, and vLLM fallback. The system integrates retrieval-augmented generation (RAG), tool calling, caching, rate limiting, and async request handling.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client (UI/HTTP)                           │
│              Streamlit Chat Interface or External API               │
└───────────────────┬─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (src/api.py)                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  POST /chat            - Non-streaming chat endpoint        │   │
│  │  POST /chat/stream     - SSE-based token streaming endpoint │   │
│  │  GET  /health          - Backend status check               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  Features:                                                          │
│  • Rate limiting (10 req/60s, in-memory deque)                      │
│  • Retry logic (up to 3 attempts with linear backoff)               │
│  • LLM response caching (InMemoryCache)                             │
│  • Multi-turn conversation history (max 20 messages)                │
│  • Request validation & error handling                              │
└───────────────────┬─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  LangGraph Pipeline (src/graph.py)                  │
│                                                                     │
│  Entry: retrieve ──► agent ──► generate ──► Output                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. RETRIEVE NODE (retrieve_node)                           │   │
│  │    • Extracts user query from last message                 │   │
│  │    • Calls Qdrant retriever with MMR (k=3)                 │   │
│  │    • Returns: retrieved_context (concatenated chunks)      │   │
│  └─────────────────┬───────────────────────────────────────────┘   │
│                    │                                                 │
│  ┌─────────────────▼───────────────────────────────────────────┐   │
│  │ 2. AGENT NODE (agent_node)                                 │   │
│  │    • Injects retrieved context as SystemMessage            │   │
│  │    • Runs ReAct agent on Groq LLM                          │   │
│  │    • Available tools:                                       │   │
│  │      - calculator: math expressions                         │   │
│  │      - doc_search: hardcoded doc lookup                     │   │
│  │      - retrieve_papers: Qdrant search (agent-callable)      │   │
│  │    • Handles tool calls & LLM reasoning loop                │   │
│  │    • Returns: new messages only (sliced to avoid dupes)     │   │
│  └─────────────────┬───────────────────────────────────────────┘   │
│                    │                                                 │
│  ┌─────────────────▼───────────────────────────────────────────┐   │
│  │ 3. GENERATE NODE (generate_node)                           │   │
│  │    • Pass-through / placeholder for future post-processing  │   │
│  │    • Can be extended for summarization, filtering, etc.     │   │
│  └─────────────────┬───────────────────────────────────────────┘   │
│                    │                                                 │
└────────────────────┼─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LLM & Orchestration Subsystem                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Primary: Groq (openai/gpt-oss-120b via langchain-groq)       │  │
│  │ • Free tier, no card required                                │  │
│  │ • Fast LLM inference via custom LPU hardware                 │  │
│  │ • Temperature: 1.0 (high variance for exploration)           │  │
│  │ • Supports function calling for tool invocation              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Fallback: vLLM (Qwen2.5-1.5B-Instruct on local GPU)          │  │
│  │ • Served via vLLM on WSL2 (RTX 3050 6GB VRAM)                │  │
│  │ • OpenAI-compatible API (ChatOpenAI interface)                │  │
│  │ • Used for resilience testing & local inference              │  │
│  │ • Not currently wired to auto-fallback chain                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  LangChain Components:                                              │
│  • ChatPromptTemplate: structured prompts with variables             │
│  • with_structured_output(): Pydantic-based response types           │
│  • create_react_agent: prebuilt ReAct loop                           │
│  • SystemMessage / HumanMessage / AIMessage: conversation turns      │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
       ┌───────────┴──────────────┬──────────────────┐
       ▼                          ▼                  ▼
┌─────────────────┐      ┌─────────────────┐  ┌──────────────────┐
│  RAG Pipeline   │      │   Vector DB     │  │  Embeddings      │
│  (src/rag.py)   │      │                 │  │                  │
└─────────────────┘      └─────────────────┘  └──────────────────┘
```

---

## Core Components

### 1. **LLM Integration** (`src/llm.py`)

**Primary LLM: Groq**
- **Model:** `openai/gpt-oss-120b` (deprecated as of Aug 2026; future updates may need version bump)
- **Interface:** `langchain_groq.ChatGroq`
- **Configuration:**
  - Temperature: 1.0 (high variance for exploration)
  - API key: `GROQ_API_KEY` (env var)
- **Capabilities:**
  - Text generation & function calling
  - Structured output via Pydantic with `with_structured_output()`
  - Message types: SystemMessage, HumanMessage, AIMessage

**Message Helpers:**
- `add_user_message()`: Append HumanMessage to list
- `add_assistant_message()`: Append AIMessage to list

**Structured Output Example:**
```python
class QueryAnalysis(BaseModel):
    topic: str
    is_question: bool
    sentiment: str  # "positive", "neutral", "negative"

structured_llm = llm.with_structured_output(QueryAnalysis)
# Returns typed Pydantic object, not raw dict/string
```

---

### 2. **Agent & Tool Calling** (`src/agent.py`)

**ReAct Agent Setup:**
```python
agent = create_react_agent(llm, tools)
```

**Built-in Tools:**

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `calculator` | Math expression string | Numeric result | Arithmetic: "15 × 7" → 105 |
| `doc_search` | Query string | Documentation snippet | Hardcoded lookup (placeholder; real RAG via graph) |
| `retrieve_papers` | Natural language query | Paper chunks + metadata | Qdrant vector search (agent-callable) |

**ReAct Loop (Automatic):**
1. LLM decides: call tool or answer directly
2. Tool execution (no recursive calling)
3. Observation returned to LLM
4. Repeat until LLM outputs final answer

---

### 3. **RAG Pipeline** (`src/rag.py`)

#### Document Loading
- **Source:** Two PDF files in `data/raw/`
  - `xvectors_speaker_recognition.pdf`
  - `ecapa_tdnn_speaker_verification.pdf`
- **Loader:** `PyPDFLoader` (LangChain community)
- **Output:** Page-level documents with source + page metadata

#### Reference Stripping
- **Problem:** Academic papers' bibliography sections scored high on content-based similarity searches, despite low relevance.
- **Solution:** Regex-based detection & truncation before chunking
  - Pattern: `(?:^|\n)\s*\d*\.?\s*(references|bibliography)\s*\n`
  - Handles numbered sections (e.g., "7. References") and page-start headings
  - Result: 68 → 54 chunks after stripping

#### Chunking
- **Method:** `RecursiveCharacterTextSplitter`
- **Settings:**
  - `chunk_size: 1000` characters
  - `chunk_overlap: 150` characters
  - Separators: `["\n\n", "\n", ". ", " ", ""]` (paragraph-aware, falls back to hard cut)
- **Output:** Semantic chunks with preserved metadata

#### Embeddings
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Reason:** Groq has no embeddings API; local embeddings used for RAG, Groq for generation
- **Dimensions:** 384
- **Inference:** CPU-based (no GPU required for 54 chunks)

#### Vector Store
- **Database:** Qdrant (Docker)
- **Collection:** `speaker_embedding_papers` (54 points after reference stripping)
- **URL:** `http://localhost:6333` (configurable via `QDRANT_URL` env var)
- **Upsert:** `QdrantVectorStore.from_documents()` (one-time, guarded against re-ingestion)

#### Retrieval
- **Search Type:** Maximal Marginal Relevance (MMR)
- **Settings:**
  - `k: 5` results returned
  - `fetch_k: 10` candidate pool
  - `lambda_mult: 0.7` (balance relevance vs. diversity)
- **Async:** Callable via `retriever.ainvoke(query)`
- **Output:** List of Document objects with content + metadata

---

### 4. **LangGraph Workflow** (`src/graph.py`)

**State Definition:**
```python
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # Accumulates messages
    retrieved_context: str                                 # Latest retrieval output
```

**Graph Structure:**

```
START
  │
  ▼
RETRIEVE NODE
  │ Retrieves chunks from Qdrant
  │ Sets retrieved_context
  ▼
AGENT NODE
  │ Injects context as SystemMessage
  │ Runs ReAct loop with tools
  │ Groq LLM decides: answer or call tool
  ▼
GENERATE NODE
  │ Pass-through (future: summarize/filter)
  ▼
END
```

**Key Behaviors:**

1. **Message Accumulation:** Messages reducer uses `operator.add`, so retrieved context is injected once at agent entry, not repeated
2. **No Tool-to-Retrieval Recursion:** Agent can call `retrieve_papers` tool directly; pipeline doesn't loop
3. **Cold-Start Ingestion Guard:**
   ```python
   if collection_exists and points_count > 0:
       use_existing_vectorstore
   else:
       load_documents → chunk → build_vectorstore
   ```
   Prevents duplicate ingestion on every `uvicorn --reload` restart

---

### 5. **FastAPI Backend** (`src/api.py`)

**Endpoints:**

| Endpoint | Method | Input | Output | Features |
|----------|--------|-------|--------|----------|
| `/health` | GET | — | `{"status": "healthy"}` | Backend liveness |
| `/` | GET | — | Service info | Root endpoint |
| `/chat` | POST | `ChatRequest` | `ChatResponse` | Non-streaming; retry logic included |
| `/chat/stream` | POST | `ChatRequest` | Server-Sent Events (SSE) | Token-by-token streaming |

**Request Models:**

```python
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []  # Prior turns (oldest first)

class ChatResponse(BaseModel):
    answer: str
    trace: list[Any]              # Full message trace
    retrieved_context: str         # Raw context passed to agent
```

#### Rate Limiting
- **Strategy:** In-memory deque of request timestamps
- **Limit:** 10 requests per 60 seconds
- **Response:** 429 Too Many Requests if exceeded
- **Scope:** Single-process; not distributed

#### Caching
- **Type:** LangChain `InMemoryCache`
- **Scope:** LLM response cache (identical queries return cached result)
- **Verification:** Repeated query runs ~90% faster than first

#### Retry Logic
- **Max Retries:** 3 attempts
- **Backoff:** Linear (1s, 2s delays between retries)
- **Scope:** Only applies to `/chat` (not `/chat/stream`)

#### Multi-Turn Conversation
- **History Window:** Max 20 prior turns
- **Mapping:** `"user"` → HumanMessage, anything else → AIMessage
- **Current Query:** Always appended as last message (so `retrieve_node` can extract it)

#### Streaming (`/chat/stream`)
- **Method:** `graph.astream_events()` with event filtering
- **Format:** Server-Sent Events (JSON payloads)
- **Filters:**
  - Only `on_chat_model_stream` events (LLM token/chunk)
  - Only from "agent" node (ignore retrieve/generate)
  - Skip tool-call chunks (empty content or `tool_call_chunks` present)
- **Payload:** `{"token": "..."}`
- **End Signal:** `{"done": true, "elapsed_s": X, "first_token_s": Y}`
- **Trade-off:** Improves time-to-first-token (~17% faster) but total time is worse (SSE overhead)

---

## Data Flow Example: "What is ECAPA-TDNN?"

### Request Phase
```
Client HTTP POST /chat
  │
  └─► ChatRequest:
        - query: "What is ECAPA-TDNN?"
        - history: []
```

### Validation & Rate Limiting
```
✓ Query not empty
✓ Rate limit check passes
✓ Input messages = [HumanMessage("What is ECAPA-TDNN?")]
```

### LangGraph Pipeline
```
1. RETRIEVE NODE
   • Input: state["messages"][-1].content = "What is ECAPA-TDNN?"
   • Retrieval: MMR search in Qdrant (k=3, fetch_k=10)
   • Output: 
     - retrieved_context = "ECAPA-TDNN is a speaker verification architecture..."
                           (chunks from papers)
   • Updated state: messages=[], retrieved_context=<text>

2. AGENT NODE
   • Input: 
     - Retrieved context (injected as SystemMessage)
     - Original query (HumanMessage)
   • LLM decision: Context contains answer → answer directly (no tool call)
   • Output:
     - new_messages = [AIMessage("ECAPA-TDNN is..."), ToolUseMessage, ...]
   • Updated state: messages=[...], retrieved_context=<unchanged>

3. GENERATE NODE
   • Input: Same state
   • Output: Pass-through (no modification)
   • Updated state: unchanged
```

### Response Generation
```
API extracts:
  • answer = last AIMessage.content
  • trace = all messages serialized (model_dump())
  • retrieved_context = state["retrieved_context"]

Returns ChatResponse:
  {
    "answer": "ECAPA-TDNN is a speaker verification...",
    "trace": [...],
    "retrieved_context": "..."
  }
```

---

## Environment Configuration

**Required env vars:**
```bash
GROQ_API_KEY=<your-groq-api-key>  # From console.groq.com (free tier)
```

**Optional env vars:**
```bash
QDRANT_URL=http://localhost:6333          # Default
VLLM_BASE_URL=http://localhost:8001/v1    # Default
```

**Example `.env`:**
```
GROQ_API_KEY=gsk_...
QDRANT_URL=http://localhost:6333
VLLM_BASE_URL=http://localhost:8001/v1
```

---

## Deployment

### Local Development

```bash
# Terminal 1: Qdrant vector DB
docker run -p 6333:6333 qdrant/qdrant

# Terminal 2: vLLM fallback (optional)
export VLLM_WSL2_ENABLE_PIN_MEMORY=1  # WSL2 only
export PYTORCH_NVML_BASED_CUDA_CHECK=0  # WSL2 only
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  # WSL2 only
export VLLM_USE_FLASHINFER_SAMPLER=0  # WSL2 only
vllm serve Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.8 --max-model-len 2048 --port 8001

# Terminal 3: FastAPI backend
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 4: Streamlit UI
streamlit run ui/streamlit_app.py
```

### Docker Compose (Full Stack)
```bash
docker-compose up -d
```
Will spin up: Qdrant, vLLM (if enabled), FastAPI, Streamlit

---

## Performance Benchmarks

### Baseline (Non-Streaming, Groq)
| Metric | Value |
|--------|-------|
| Mean latency | 3.77s |
| Median (p50) | 3.65s |
| p95 | 4.84s |
| Throughput | 0.27 req/s (sequential) |

### Streaming (SSE, per token)
| Metric | Value |
|--------|-------|
| Time to first token (p50) | 3.02s |
| Total completion time (mean) | 5.18s |
| Improvement (TTFT) | ~17% faster |

---

## Known Limitations & Future Work

1. **No Distributed Caching:** InMemoryCache is single-process only
2. **Streaming Retry Gap:** `/chat/stream` lacks retry logic (would duplicate content)
3. **Retrieval Quality:** MMR works well for speaker embedding papers but not tuned for arbitrary domains
4. **No Persistent Chat History:** Conversation stored only in client/RAM
5. **vLLM Cold Start:** ~5-10 min first-time startup on 6GB GPU

---

## Testing

**Sanity Checks:**
```bash
python -m pytest tests/
```

**LLM Integration Test:**
```bash
python src/llm.py
```

**RAG Pipeline Test:**
```bash
python src/rag.py
# Outputs retrieval results for test queries
```

**LangGraph Async Test:**
```bash
python src/graph.py
# Outputs agent trace & final response
```

**API Test:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is statistics pooling?", "history": []}'
```

---

## Glossary

- **RAG:** Retrieval-Augmented Generation — fetch context before generation
- **ReAct:** Reasoning + Acting — LLM thinks through steps & calls tools
- **MMR:** Maximal Marginal Relevance — retrieve diverse, relevant results
- **LangGraph:** Graph-based workflow orchestration (LangChain)
- **Groq:** LLM provider with fast inference via LPU hardware
- **Qdrant:** Open-source vector database for embeddings
- **vLLM:** Open-source inference engine with efficient KV cache
- **SSE:** Server-Sent Events — HTTP streaming protocol
- **ChainSame:** (Assistant terminology) conversation turns persist across requests

