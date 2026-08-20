# RAG Assistant

An AI assistant built with **Google Gemini** as the primary LLM, orchestrated through **LangChain + LangGraph**, with **RAG** over a Qdrant vector store, tool-calling, and a **local vLLM** model as an automatic fallback provider.

Built as a two-part assignment: (1) an applied AI assistant with RAG and tool use, and (2) a productionized version with async handling, caching, retries, rate limiting, and Docker deployment.

## Status
🚧 In progress — see [`NOTES.md`](./NOTES.md) for the running build log.

## Features
- **LLM integration:** Gemini (`gemini-2.5-flash`) via `langchain-google-genai`, with configurable temperature/top_p
- **Structured output:** typed responses via `with_structured_output`
- **Tool calling:** LangGraph `create_react_agent` with custom tools
- **RAG pipeline:** document ingestion → chunking → Gemini embeddings → Qdrant vector store → retriever
- **Local model:** vLLM-served open-source model, used both as a standalone deliverable and as an automatic fallback if Gemini fails or rate-limits
- **Web UI:** Streamlit chat interface showing intermediate agent steps (tool calls, retrieved chunks)
- **Reliability:** retries (`.with_retry()`), fallback provider (`.with_fallbacks()`), rate limiting, graceful degradation
- **Performance:** async request handling, response caching, streaming
- **Deployment:** Dockerized, with Docker Compose for the full stack (backend, UI, Qdrant, vLLM)

## Tech stack
| Layer | Choice |
|---|---|
| LLM (primary) | Google Gemini via `langchain-google-genai` |
| LLM (fallback / local) | vLLM-served open-source model via `langchain-openai` |
| Orchestration | LangChain + LangGraph |
| Vector DB | Qdrant |
| Embeddings | Gemini embeddings (`models/text-embedding-004`) — same provider as chat, no separate embedding model needed |
| Backend | FastAPI |
| UI | Streamlit |
| Deployment | Docker / Docker Compose |

## Setup

```bash
# clone and enter the repo
git clone https://github.com/<your-username>/gemini-rag-assistant.git
cd gemini-rag-assistant

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# set up environment variables
cp .env.example .env
# then edit .env and add your GOOGLE_API_KEY (from Google AI Studio, no card required)
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
gemini-rag-assistant/
├── src/                  # backend application code
│   ├── llm.py            # Gemini + local vLLM client setup
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
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── NOTES.md
└── README.md
```

## Notes on scope
This project runs on Google Gemini's free tier (no payment method required) rather than a paid provider, with a local vLLM-served model standing in as the fallback/secondary provider. An earlier build attempt targeted xAI's Grok API, but that account had no free credits available at the time (see `NOTES.md` for details) — Gemini was chosen as the replacement because its free tier covers both chat and embeddings under one provider, removing the need for a separate embeddings workaround.

## License
*(add if required by your assignment)*
