# RAG Assistant

An AI assistant built with **Groq** as the primary LLM, orchestrated through **LangChain + LangGraph**, with **RAG** over a Qdrant vector store, tool-calling, and a **local vLLM** model as an automatic fallback provider.

Built as a two-part assignment: (1) an applied AI assistant with RAG and tool use, and (2) a productionized version with async handling, caching, retries, rate limiting, and Docker deployment.

## Status
🚧 In progress — see [`NOTES.md`](./NOTES.md) and [`context.md`](./context.md) for the running build log and decision history.

## Features
- **LLM integration:** Groq (`llama-3.3-70b-versatile`) via `langchain-groq`, with configurable temperature/top_p
- **Structured output:** typed responses via `with_structured_output`
- **Tool calling:** LangGraph `create_react_agent` with custom tools
- **RAG pipeline:** document ingestion → chunking → local embeddings → Qdrant vector store → retriever
- **Local model:** vLLM-served open-source model, used both as a standalone deliverable and as an automatic fallback if Groq fails or rate-limits
- **Web UI:** Streamlit chat interface showing intermediate agent steps (tool calls, retrieved chunks)
- **Reliability:** retries (`.with_retry()`), fallback provider (`.with_fallbacks()`), rate limiting, graceful degradation
- **Performance:** async request handling, response caching, streaming
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
git clone https://github.com/<your-username>/groq-rag-assistant.git
cd groq-rag-assistant

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
*(to be filled in as the backend and UI come online — see the roadmap in `docs/`)*

```bash
# run the backend
uvicorn src.api:app --reload

# run the UI (separate terminal)
streamlit run ui/streamlit_app.py
```

## Architecture
See [`docs/architecture.md`](./docs/architecture.md) for the diagram and request-flow breakdown (added once the pipeline is wired end-to-end).

## Project structure
```
groq-rag-assistant/
├── src/                  # backend application code
│   ├── llm.py            # Groq + local vLLM client setup
│   ├── agent.py           # LangGraph tool-calling agent
│   ├── rag.py             # ingestion, chunking, retrieval
│   ├── cache.py           # response caching
│   └── api.py             # FastAPI app
├── ui/
│   └── streamlit_app.py   # chat UI
├── data/
│   └── raw/                # source documents for RAG
├── docs/
│   └── architecture.md     # architecture diagram + notes
├── tests/
│   └── sanity_check.py     # environment/API verification script
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── NOTES.md
├── context.md               # decision log / project history
└── README.md
```

## Notes on scope
This project runs on Groq's free tier rather than a paid provider, with a local vLLM-served model standing in as the fallback/secondary provider. Groq was reached after evaluating and ruling out two other providers first: xAI's Grok API (no free credits available on the account used) and Google Gemini (a persistent, widely-reported free-tier access error unrelated to this project's configuration). See [`context.md`](./context.md) for the full reasoning behind each pivot.

