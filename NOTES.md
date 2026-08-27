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