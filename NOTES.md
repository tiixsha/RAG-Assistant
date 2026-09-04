## LLM Integration + Prompt Engineering
- Studied Claude Learning Path (API basics, messages) + Prompt Engineering Guide (system prompts, temp/top_p, few-shot). Concept translated cleanly to LangChain — SystemMessage/HumanMessage/AIMessage map directly onto Claude's message roles.
- Wired up ChatGroq (model: openai/gpt-oss-120b — note: llama-3.3-70b-versatile is deprecated as of Aug 2026), confirmed llm.invoke() returns text via sanity_check.py.
- Built add_user_message/add_assistant_message helpers around LangChain message objects.
- Built system prompt with ChatPromptTemplate instead of string concat; chained with `prompt | llm`.
- Compared temperature 0.2 / 0.7 / 1.2 on the same prompt — lower temp gave safe/repetitive phrasing, higher temp gave more varied wording, as expected.
- Added structured output via `llm.with_structured_output(QueryAnalysis)` — confirmed with isinstance() that output is a typed Pydantic object, not raw text/dict.
- Gotcha hit: `from itertools import chain` shadowed the imported `chain` from src.llm — reminder to watch for stdlib name collisions.

## Tool Calling
- Studied Claude Learning Path's tool use module (concept) + LangGraph's create_react_agent docs — prebuilt agent replaces most of the hand-rolled reason/act/observe loop.
- Built 2 tools with @tool decorator in src/agent.py: calculator (eval-based, noted as fine for this assignment but not production-safe) and doc_search (hardcoded lookup, stand-in until Day 3-4's real Qdrant retriever exists).
- Wired both into a LangGraph agent via create_react_agent(llm, tools).
- Verified 3 behaviors: (1) single tool call — "15 × 7" correctly invoked calculator, returned 105; (2) no unnecessary tool call — "capital of France" answered directly with tool_calls=[]; (3) chained tool calls — "search groq docs, then multiply 8 by word count" correctly called doc_search first, reasoned over the result (counted 16 words), then called calculator with 8*16, returning 128.
- Confirmed openai/gpt-oss-120b supports function calling cleanly via Groq — no issues.
- Gotcha hit: ModuleNotFoundError: No module named 'src' when running sanity_check.py — the sys.path.append fix from Day 1 was missing from the rewritten test file. Re-added it; worth keeping that boilerplate at the top of every test file going forward.
- Full message trace from the chained-tool-call test is a good candidate for the Day 6 architecture diagram — shows the real reasoning → tool → observation loop end to end.

## Ingestion & Chunking

Loaded 2 source PDFs (xvectors_speaker_recognition.pdf, ecapa_tdnn_speaker_verification.pdf)
via PyPDFLoader — 10 pages total, 68 chunks after splitting. All chunks verified to carry
source + page metadata.

Chunking: RecursiveCharacterTextSplitter, chunk_size=1000, chunk_overlap=150,
separators=[\n\n, \n, ". ", " ", ""] (paragraph/sentence-aware, falls back to hard cut).
Reasonable starting point for dense technical PDF content — not tuned yet, will revisit
if Day 4 retrieval tests surface fragmented or noisy hits.

## Embeddings & Vector DB

Set up Qdrant via Docker, embedded chunks locally with HuggingFaceEmbeddings
(sentence-transformers/all-MiniLM-L6-v2) — Groq has no embeddings API, so chat (Groq)
and embeddings (local) are two separate models for two separate jobs, documented as
a deliberate architecture choice, not a compromise. Upserted into Qdrant via
QdrantVectorStore.from_documents(), wrapped as a retriever (k=3), tested with 3
sample queries against the actual corpus (x-vectors extraction, ECAPA-TDNN
architecture, evaluation methodology).

Finding: initial retrieval was pulling bibliography/references chunks into top-k
results (e.g. citation lists like "[1] D. Snyder... X-vectors: Robust DNN embeddings"
scoring high on an x-vector query, purely from keyword/topic overlap in citation
titles, not actual content relevance).

Fix: added strip_references() in rag.py — detects each paper's References/
Bibliography heading via regex and truncates that page's content there, dropping
all subsequent pages before chunking. Chose this over a per-chunk citation-density
heuristic: it fixes the problem structurally (using document structure) rather than
probabilistically filtering symptoms after the fact, avoids threshold-tuning, and
generalizes to any future academic-paper source.

Regex needed two iterations to handle real-world heading variance:
1. `\n\s*(references|bibliography)\s*\n` — missed ECAPA's "7. References" (numbered
   heading) and a heading at the very start of a page (no leading \n to anchor on).
2. Final: `(?:^|\n)\s*\d*\.?\s*(references|bibliography)\s*\n` — handles optional
   section numbering and start-of-page headings.

Result: chunk count 68 -> 60 (x-vectors stripped) -> 54 (ECAPA stripped, after regex
fix). Reran all 3 test queries post-fix: no bibliography chunks in any top-3 result.
Query 3 (evaluation) hits are now real content but somewhat generic/introductory —
flagged as an area to revisit if retrieval quality matters more in a later day, not
blocking for Day 4's scope.

## Local Deployment with vLLM (WSL2 + RTX 3050 6GB)

Goal: serve a small quantized model locally via vLLM, exposed through the same
LangChain ChatOpenAI interface as Groq (per roadmap: one-line swap for Day 9's
fallback chaining). Chose Qwen2.5-1.5B-Instruct (fp16, not quantized) as a
hardware-driven simplification for 6GB VRAM — documenting per roadmap's own
"CPU + tiny model is fine... document as hardware-constrained simplification"
guidance.

vLLM has no native Windows support; ran via WSL2 (Ubuntu 22.04) instead of
bare Windows. This surfaced a chain of environment issues, each fixed in turn:

1. RuntimeError: UVA is not available — known vLLM-on-WSL2 GitHub issue (GPU
   passthrough works, but CUDA Unified Virtual Addressing isn't fully
   supported under WSL2's passthrough). Fix: export VLLM_WSL2_ENABLE_PIN_MEMORY=1

2. Failed to find C compiler / Python.h missing — WSL's minimal install
   lacked build-essential (gcc) initially, then lacked python3.10-dev
   (Python's C headers, a separate package from the interpreter). Both
   needed for Triton's runtime JIT kernel compilation.
   Fix: sudo apt install build-essential python3.10-dev -y

3. Can't initialize NVML / 0 active drivers found — after a WSL restart,
   PyTorch's own NVML binding failed even though system-level `nvidia-smi`
   worked fine. Root cause: PyTorch defaults to an NVML-based CUDA check
   that's flaky under WSL2's driver passthrough.
   Fix: export PYTORCH_NVML_BASED_CUDA_CHECK=0
        export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

4. ValueError: No available memory for the cache blocks (KV cache = negative) —
   --gpu-memory-utilization 0.5 (3GB of 6GB) wasn't enough once model weights
   (2.98 GiB) + CUDA context + activation memory were accounted for, leaving
   nothing for the KV cache pool itself.
   Fix: raised --gpu-memory-utilization to 0.8

5. RuntimeError: Could not find nvcc / cuda_home doesn't exist — FlashInfer's
   fast sampling kernel needs the full CUDA toolkit (nvcc) to JIT-compile;
   WSL only had the pip-installed CUDA runtime libs, not the toolkit.
   Fix: export VLLM_USE_FLASHINFER_SAMPLER=0 (falls back to vLLM's built-in
   PyTorch-native sampler — correctness unaffected, small speed cost only)

Final working launch command (env vars must be re-exported every new shell
session, they don't persist):
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
  export PYTORCH_NVML_BASED_CUDA_CHECK=0
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  export VLLM_USE_FLASHINFER_SAMPLER=0
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.8 --max-model-len 2048 --port 8001

Verified working: /v1/models and /v1/chat/completions both return correctly
via curl. Connected via LangChain's ChatOpenAI(base_url="http://localhost:8001/v1",
model="Qwen/Qwen2.5-1.5B-Instruct") in rag.py — same interface as ChatGroq,
confirming the roadmap's Day 9 fallback-chaining premise holds.

Startup is slow (~5-10 min cold: model download/load + torch.compile +
CUDA graph capture), largely unavoidable on this hardware — first
torch.compile pass alone took 102s. Compile artifacts cache to disk after
a successful run, so subsequent starts load faster.

## Pipeline Wire-Up

Assembled the full LangGraph pipeline in src/graph.py: retrieve -> agent -> generate,
per roadmap. Key pieces:
- retrieve_node: upfront Qdrant search against the user's query
- agent_node: ReAct agent (calculator, doc_search, and a new retrieve_papers tool
  backed by the same Qdrant retriever), grounded via a SystemMessage injecting the
  retrieved context before the agent runs
- generate_node: pass-through, kept as a distinct node for the diagram and as a
  future post-processing hook (Day 9+)

Two real bugs hit and fixed during this step:
1. Message duplication — GraphState used operator.add as the messages reducer, but
   create_react_agent's invoke() returns full conversation history (including the
   echoed human message), not just new turns. Returning result["messages"] wholesale
   double-added the human message. Fixed by slicing to only the new messages:
   result["messages"][len(input_messages):]
2. Retrieval never actually used — retrieve_node computed retrieved_context but
   agent_node never passed it to the agent, so the LLM answered from its own
   training knowledge instead of the retrieved paper content (confirmed by checking
   whether answers cited real ablation-table numbers vs. generic ML knowledge).
   Fixed by injecting retrieved_context as a SystemMessage before the agent runs.
   Verified fix: reran the ECAPA-TDNN test query, answer now cites actual paper
   content (ablation rows B.1/B.2/C.1-3, EER/MinDCF figures from the real table).

Also made QDRANT_URL and vLLM's base_url configurable via environment variables
(os.getenv with localhost defaults) in rag.py and graph.py, so a future Docker/
Compose setup can point them at container-network addresses without code changes.

## Retrieval Improvements & Streamlit Validation

Improved the Qdrant retrieval strategy after inspecting the Day 4-6 retrieval results. The previous retriever used basic similarity search with k=3, which sometimes returned highly similar or less useful chunks even after the bibliography/reference filtering fix.

Updated the retriever in `src/rag.py` to use Maximal Marginal Relevance (MMR):

```python
def get_retriever(vectorstore, k: int = 5):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10,
            "lambda_mult": 0.7,
        },
    )
```

MMR was chosen to improve result diversity while retaining semantic relevance. Retrieval now considers a larger candidate pool (`fetch_k=10`) before selecting the final 5 chunks.

Also increased the displayed retrieval preview from 120 to 500 characters to make manual inspection of retrieved context more useful.

Verified the Qdrant collection after ingestion: `speaker_embedding_papers` contains exactly 54 points, matching the expected 54 chunks. No duplicate-ingestion issue was present, so no further changes to the ingestion pipeline were required.

Reran retrieval tests against the actual corpus using:

1. "How does the x-vector system extract speaker embeddings?"
2. "What is the ECAPA-TDNN architecture?"
3. "How is speaker verification evaluated in these papers?"
4. "What is statistics pooling in the x-vector system?"
5. "What are the main components of ECAPA-TDNN?"

MMR retrieval successfully returned relevant paper content. In particular, the statistics-pooling query retrieved the x-vector section describing how mean and standard deviation are computed over frame-level outputs and aggregated across the time dimension.

The x-vector extraction query also retrieved the relevant section describing extraction of the speaker embedding from the segment-level network.

Some queries still returned generic or less relevant chunks alongside useful results, so retrieval quality is considered functional but not fully optimized. This is not blocking the current pipeline.

### Streamlit UI Testing

Moved to end-to-end validation through the Streamlit interface after confirming that ingestion, Qdrant, retrieval, and LangGraph integration are working.

The Streamlit testing focus is on exposing:

* User query and generated response
* Retrieved document context
* Source PDF and page metadata
* Full LangGraph message trace
* Intermediate tool/agent execution

The LangGraph pipeline remains in-process for this stage. HTTP/FastAPI integration is intentionally deferred to Day 8.

Validation queries planned for the Streamlit interface:

* "How does the x-vector system extract speaker embeddings?"
* "What is statistics pooling in the x-vector system?"
* "What is 15 × 7?" — validates calculator/tool execution and trace visibility.
* "What is the capital of France?" — validates handling of unrelated questions without unnecessarily invoking retrieval/tools.



## Async Backend & Caching 

Wrapped the LangGraph pipeline in FastAPI (`src/api.py`). `/chat` accepts a
single `query` string and calls `await graph.ainvoke(...)`, converting the
graph's async node functions (already `async def` in `graph.py`) into a real
async request path end to end.

Added LLM response caching via `set_llm_cache(InMemoryCache())`. Verified by
timing two identical requests back to back — second request completed
substantially faster than the first, confirming the cache is actually being
hit rather than just configured.

Concurrency tested with `tests/test_async.py` sending 5 distinct queries
concurrently (deliberately different questions, not repeats — a repeated
query would just test the cache, not real concurrency). All 5 completed
successfully; FastAPI terminal logged 5 separate `/chat` requests as
expected.

Gotcha hit: original `api.py` draft had a duplicate `from src.graph import
graph` import — harmless but sloppy; cleaned up in the rewrite.

## Reliability: Retry & Rate Limiting 

Added retry logic around `graph.ainvoke()` in `/chat`: up to 3 attempts with
linear backoff (1s, 2s) before returning a 500. Verified the normal path
still returns 200 after adding the retry wrapper (retry logic shouldn't
change behavior when nothing's actually failing).

Added a simple in-memory rate limiter (`deque` of request timestamps, 10
requests / 60s window) rather than pulling in Redis or another distributed
component — appropriately scoped for this project's size. Verified by
sending 12 requests in quick succession; requests beyond the 10th correctly
returned `429 Too Many Requests` with a clear `detail` message.

Verified 3 concrete test cases:
1. Normal query → 200 OK
2. Empty query (`""`) → 400 Bad Request, `"Query cannot be empty."`
3. 11th+ request within 60s → 429 Too Many Requests

**Scoping decision:** the roadmap's third Day 9 reliability piece,
`.with_fallbacks()` chaining Groq → local vLLM, was not implemented in this
pass — only retry-on-the-same-provider and rate limiting were built. Noting
this explicitly as a deliberate scope decision, not an oversight, in case
Groq-failure resilience needs revisiting later.

## Bugfix: Qdrant Duplicate-Points on Every Reimport

Found during Day 8/9 testing: `graph.py`'s module-level ingestion
(`build_vectorstore(_chunks, recreate=False)`) ran on every import of
`src.graph` — including every `uvicorn --reload` restart during active
development. `recreate=False` only skips *deleting* the collection; it does
NOT skip re-embedding and re-upserting. Repeated restarts silently
multiplied the collection: caught it at 486 points (9× the correct 54),
confirmed via `curl http://localhost:6333/collections/speaker_embedding_papers`.

Fix: guarded the ingestion block —
```python
_collection_ready = (
    _client.collection_exists(COLLECTION_NAME)
    and _client.get_collection(COLLECTION_NAME).points_count > 0
)
if _collection_ready:
    _vectorstore = QdrantVectorStore(
        client=_client, collection_name=COLLECTION_NAME, embedding=get_embeddings(),
    )
else:
    _documents = load_documents()
    _chunks = chunk_documents(_documents)
    _vectorstore = build_vectorstore(_chunks, recreate=False)
```
Verified fix across two consecutive `uvicorn --reload` restarts: startup log
correctly printed `"already populated — skipping re-ingestion"` both times,
and `points_count` held steady at 54 rather than climbing.

## Streamlit Over HTTP

Rewrote `ui/streamlit_app.py` to remove the in-process `from src.graph
import graph` entirely, replacing `pipeline.invoke(...)` with
`requests.post()` against `/chat`. This closes out the piece of Day 7's
checkpoint that was deliberately deferred back when the UI was first built
("HTTP wiring lands once Day 8's FastAPI wrapper exists").

Added explicit UI handling for the new failure modes FastAPI can now return:
429 (rate limited), 400 (empty query), connection errors (backend not
running), and timeouts — none of which existed as concepts in the
in-process version.

Trace panel updated to read plain dicts (`.get()`) instead of LangChain
message objects (`getattr()`), since `/chat` serializes messages with
`model_dump()`/`.dict()` before returning them.

Added a `/health`-backed backend-status indicator in the sidebar.

## Baseline (non-streaming, Groq via /chat)
| Metric | Value |
|---|---|
| Mean latency | 3.77s |
| p50 | 3.65s |
| p95 | 4.84s |
| Min / Max | 2.96s / 4.84s |
| Throughput | 0.27 req/s (sequential) |

### ONNX Export Attempt
Attempted on the local vLLM model (Qwen/Qwen2.5-1.5B-Instruct) via
optimum.exporters.onnx. Note: roadmap referenced this module path from an
older optimum version — by mid-2026 Hugging Face split ONNX export into a
separate `optimum-onnx` package; the original module path still worked once
installed correctly. Export completed successfully (`onnx_qwen/`), but
validation flagged numerical drift exceeding the default 1e-05 tolerance
across all 28 transformer layers' KV-cache outputs (max diff ~0.001) —
consistent with typical fp16/cross-framework floating-point accumulation in
autoregressive LLM export, not a structural failure. Model is usable but
would need relaxed tolerance or explicit accuracy testing before treating
it as production-equivalent. Not pursued further — vLLM remains the actual
serving path; this satisfied the roadmap's ONNX checkpoint only.

### Optimization: SSE Streaming (/chat/stream)
Added a second endpoint streaming tokens via graph.astream_events(),
targeting perceived latency (time-to-first-token) rather than total
throughput.

| Metric | Baseline (non-streaming) | Streaming |
|---|---|---|
| p50 first-response time | 3.65s | 3.02s (first token) |
| Mean | 3.77s | 4.40s (first token, skewed by 1 cold-start outlier) |
| Mean total completion | 3.77s | 5.18s |

Streaming improves median perceived latency by
~17% (3.65s → 3.02s to first visible output) but does NOT reduce total
generation time — total completion is measurably worse (5.18s vs 3.77s
mean), due to per-event/per-chunk SSE overhead inherent to token-level
streaming via astream_events(), not any tool-routing inefficiency (verified:
added node/tool-call-chunk filters had zero effect on the numbers, confirming
no tool-call chunks were present — these test queries triggered a single
LLM call per turn, not the two-call ReAct pattern originally suspected).
This is a legitimate trade-off, not a clean win: better for interactive UX,
worse for raw throughput. /chat/stream also does not include Day 9's retry
logic — retrying after partial output has streamed would mean duplicating
or truncating content, so this is a deliberate reliability gap for this
endpoint, not an oversight.

### PyTorch DDP
Not applicable — this project serves models (Groq API, local vLLM), it does
not train them. No distributed training component exists to apply DDP to.