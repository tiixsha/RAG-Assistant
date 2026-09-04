# RAG Assistant

An AI assistant built with **Groq** as the primary LLM, orchestrated through **LangChain + LangGraph**, with **RAG** over a Qdrant vector store, tool-calling, and a **local vLLM** model as an automatic fallback provider.

Built as a two-part assignment: (1) an applied AI assistant with RAG and tool use, and (2) a productionized version with async handling, caching, retries, rate limiting, and Docker deployment.

## Status
✅ Complete — both Task 1 (Applied AI Assistant) and Task 2 (Productionization) deliverables are implemented. 

## Features
- **LLM integration:** Groq (`llama-3.3-70b-versatile`) via `langchain-groq`, with configurable temperature/top_p
- **Structured output:** typed responses via `with_structured_output`
- **Tool calling:** LangGraph `create_react_agent` with custom tools
- **RAG pipeline:** document ingestion → chunking → local embeddings → Qdrant vector store → retriever
- **Local model:** vLLM-served open-source model, used both as a standalone deliverable and as an automatic fallback if Groq fails or rate-limits
- **Web UI:** Streamlit chat interface (`ui/streamlit_app.py`) showing intermediate agent steps (tool calls, retrieved chunks)
- **Reliability:** retries (`.with_retry()`), fallback provider (`.with_fallbacks()`), rate limiting, graceful degradation
- **Performance:** async request handling, response caching, streaming, with dedicated latency/throughput benchmark scripts
- **Deployment:** Dockerized, with Docker Compose for the full stack (backend, UI, Qdrant, vLLM)

## Tech stack
| Layer | Choice |
|---|---|
| LLM (primary) | Groq via `langchain-groq` |
| LLM (fallback / local) | vLLM-served open-source model via `langchain-openai` |
| Orchestration | LangChain + LangGraph |
| Vector DB | Qdrant |
| Embeddings | Local (HuggingFace sentence-transformers) — Groq has no embeddings API |
| Backend | FastAPI |
| UI | Streamlit |
| Deployment | Docker / Docker Compose |

## Setup

```bash
# clone and enter the repo
git clone https://github.com/tiixsha/RAG-Assistant.git
cd RAG-Assistant

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# set up environment variables
cp .env.example .env
# then edit .env and add your GROQ_API_KEY (from console.groq.com, no card required)
```

## Usage

### Run with Docker Compose
```bash
docker compose up
```

### Run locally 
```bash
# start the backend
python src/api.py
# or: uvicorn src.api:app --reload

# in a separate terminal, run the UI
streamlit run ui/streamlit_app.py
```

Confirm the backend is up:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "What is 2+2?"}'
```

This starts the backend, UI, Qdrant, and the local vLLM service together. The `Dockerfile` and `docker-compose.yml` are complete and build correctly; on this development machine, the final container run hit a known Docker Desktop / WSL2 disk I/O slowdown during image export that didn't resolve within the assignment window, so the Docker path was verified up through image build rather than a full live `docker compose up` demo. The local run above is the tested, working path and uses the same codebase.

### Benchmarks
```bash
python tests/benchmark_latency.py
python tests/benchmark_streaming.py
```
Latency/throughput figures (p50, p95) measured against Groq are logged in `NOTES.md`.

## Architecture
See [`docs/architecture.md`](./docs/architecture.md) and the accompanying diagram in `docs/` for the full request-flow breakdown.

Request flow: Streamlit UI → FastAPI backend → response cache → LangGraph agent (`graph.py`, Groq primary with automatic fallback to local vLLM via `.with_fallbacks()`) → retriever → Qdrant (embeddings generated locally via HuggingFace sentence-transformers).

## Project structure
```
RAG-Assistant/
├── data/
│   └── raw/                     # source documents for RAG
├── docs/
│   ├── architecture.md          # architecture notes
│   └── architecture_diagram.png # architecture diagram
├── src/
│   ├── __init__.py
│   ├── agent.py                 # LangGraph tool-calling agent
│   ├── api.py                   # FastAPI app
│   ├── cache.py                 # response caching
│   ├── graph.py                 # full LangGraph pipeline assembly
│   ├── llm.py                   # Groq + local vLLM client setup
│   └── rag.py                   # ingestion, chunking, retrieval
├── tests/
│   ├── benchmark_latency.py     # latency benchmarking (Day 10)
│   ├── benchmark_streaming.py   # streaming throughput benchmarking
│   ├── sanity_check.py          # environment/API verification script
│   └── test_async.py            # async/concurrency tests
├── ui/
│   ├── env.example
│   └── streamlit_app.py         # chat UI
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── NOTES.md
└── README.md
```

**Not committed (local-only, excluded via `.gitignore`):**
- `myenv/`, `vllm-env/` — Python virtual environments
- `__pycache__/` — compiled bytecode
- `.env` — contains the real API key
- `qdrant_storage/` — Qdrant's local runtime data, regenerated on startup
- `onnx_qwen/` — exported ONNX model artifact; excluded for size, the export step itself is documented in the code and in `NOTES.md`

