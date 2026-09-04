"""
src/api.py — Async FastAPI backend with caching, retry logic, rate limiting,
and multi-turn conversation history support.
"""

from pathlib import Path
import sys
import time
import asyncio
from collections import deque
from typing import Any

# -------------------------------------------------------------------
# Project path
# -------------------------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


# -------------------------------------------------------------------
# External imports
# -------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from src.graph import graph


# -------------------------------------------------------------------
# LLM CACHE
# -------------------------------------------------------------------

set_llm_cache(
    InMemoryCache()
)


# -------------------------------------------------------------------
# RATE LIMITING
# -------------------------------------------------------------------

RATE_LIMIT = 10
RATE_WINDOW = 60
REQUEST_TIMESTAMPS = deque()


def check_rate_limit():
    """
    Simple in-memory limiter: 10 requests per 60 seconds.
    Suitable for this project's scale, not a distributed deployment.
    """

    now = time.time()

    while REQUEST_TIMESTAMPS:
        oldest_request = REQUEST_TIMESTAMPS[0]
        if now - oldest_request > RATE_WINDOW:
            REQUEST_TIMESTAMPS.popleft()
        else:
            break

    if len(REQUEST_TIMESTAMPS) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )

    REQUEST_TIMESTAMPS.append(now)


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="Speaker Embedding AI Assistant",
    description=(
        "Async LangGraph RAG backend "
        "with caching, reliability features, and conversation history"
    ),
    version="1.1.0",
)


# -------------------------------------------------------------------
# Request / Response models
# -------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single turn of prior conversation history sent by the client."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    # Prior conversation turns, oldest first. Does NOT include the current
    # `query` itself — the client sends that separately. Defaults to empty
    # for backward compatibility with any caller that only sends `query`.
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    trace: list[Any]
    retrieved_context: str


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

# Cap how much prior history gets forwarded to the graph. Unbounded history
# would grow the prompt indefinitely across a long session; this is a simple
# fixed window rather than real summarization/truncation-by-tokens.
MAX_HISTORY_MESSAGES = 20


def history_to_lc_messages(history: list[ChatMessage]) -> list:
    """
    Convert client-supplied {role, content} history into LangChain message
    objects, same mapping the original in-process Streamlit UI used via its
    to_lc_messages() helper: "user" -> HumanMessage, anything else -> AIMessage.
    """
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    lc_messages = []
    for turn in trimmed:
        if turn.role == "user":
            lc_messages.append(HumanMessage(content=turn.content))
        else:
            lc_messages.append(AIMessage(content=turn.content))
    return lc_messages


# -------------------------------------------------------------------
# Health endpoints
# -------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Speaker Embedding AI Assistant",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }


# -------------------------------------------------------------------
# Chat endpoint
# -------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):

    # ---------------------------------------------------------------
    # Validate query
    # ---------------------------------------------------------------

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # ---------------------------------------------------------------
    # Rate limiting
    # ---------------------------------------------------------------

    check_rate_limit()

    # ---------------------------------------------------------------
    # Build input messages: prior history + current query.
    # retrieve_node reads state["messages"][-1] as the retrieval query,
    # so the current turn must end up last in this list.
    # ---------------------------------------------------------------

    input_messages = history_to_lc_messages(request.history) + [
        HumanMessage(content=query)
    ]

    # ---------------------------------------------------------------
    # Start request timer
    # ---------------------------------------------------------------

    start = time.perf_counter()

    # ---------------------------------------------------------------
    # Retry configuration
    # ---------------------------------------------------------------

    max_retries = 3
    result = None

    for attempt in range(max_retries):

        try:
            print(
                f"Processing request "
                f"(attempt {attempt + 1}/{max_retries}, "
                f"history turns: {len(request.history)})"
            )

            result = await graph.ainvoke(
                {
                    "messages": input_messages,
                    "retrieved_context": "",
                }
            )

            break

        except Exception as exc:

            print(
                f"Attempt {attempt + 1}/"
                f"{max_retries} failed: {exc}"
            )

            if attempt == max_retries - 1:

                import traceback
                print("\n--- REQUEST FAILED ---")
                traceback.print_exc()

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "The AI service failed after "
                        "multiple attempts."
                    ),
                )

            await asyncio.sleep(attempt + 1)

    # ---------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------

    if result is None:
        raise HTTPException(
            status_code=500,
            detail="No result was returned by the graph.",
        )

    # ---------------------------------------------------------------
    # Extract messages
    # ---------------------------------------------------------------

    messages = result.get("messages", [])

    # ---------------------------------------------------------------
    # Extract final AI answer
    # ---------------------------------------------------------------

    answer = ""

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            content = message.content
            if isinstance(content, str):
                answer = content
                break

    if not answer:
        answer = "No answer was generated."

    # ---------------------------------------------------------------
    # Serialize message trace
    # ---------------------------------------------------------------

    trace = []

    for message in messages:
        if hasattr(message, "model_dump"):
            trace.append(message.model_dump())
        elif hasattr(message, "dict"):
            trace.append(message.dict())
        elif isinstance(message, dict):
            trace.append(message)
        else:
            trace.append(
                {
                    "type": getattr(message, "type", "unknown"),
                    "content": getattr(message, "content", ""),
                }
            )

    # ---------------------------------------------------------------
    # Retrieved context
    # ---------------------------------------------------------------

    retrieved_context = result.get("retrieved_context", "")

    # ---------------------------------------------------------------
    # Request timing
    # ---------------------------------------------------------------

    elapsed = time.perf_counter() - start

    print(f"Request completed in {elapsed:.2f}s")

    # ---------------------------------------------------------------
    # Return response
    # ---------------------------------------------------------------

    return ChatResponse(
        answer=answer,
        trace=trace,
        retrieved_context=retrieved_context,
    )


# -------------------------------------------------------------------
# Streaming chat endpoint (Day 10 optimization)
# -------------------------------------------------------------------
# Uses graph.astream_events() to stream LLM tokens as Server-Sent Events
# (SSE) as soon as they're generated, rather than waiting for the full
# response like /chat does. This targets time-to-first-token, not total
# generation time — the LLM still has to generate the same number of
# tokens either way, but the user sees output starting almost immediately
# instead of waiting the full ~3-4s baseline before seeing anything.
#
# Deliberately does NOT include /chat's retry logic: retrying after
# partial output has already been streamed to the client would mean
# either duplicating tokens or silently truncating what the user already
# saw. Retry-then-stream is a real gap for production use but out of
# scope for this pass — noted here rather than silently omitted.
# -------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    check_rate_limit()

    input_messages = history_to_lc_messages(request.history) + [
        HumanMessage(content=query)
    ]

    async def event_generator():
        start = time.perf_counter()
        first_token_time = None
        chunk_count = 0
        skipped_tool_chunks = 0

        try:
            async for event in graph.astream_events(
                {
                    "messages": input_messages,
                    "retrieved_context": "",
                },
                version="v2",
            ):
                # "on_chat_model_stream" fires once per generated token/chunk
                # from the underlying ChatGroq model, including chunks
                # produced inside the create_react_agent subgraph.
                if event["event"] != "on_chat_model_stream":
                    continue

                chunk = event["data"].get("chunk")
                if chunk is None:
                    continue

                # Filter 1: only forward events emitted from within the
                # "agent" node. retrieve_node/generate_node never call the
                # LLM in the current graph, so this is currently a no-op
                # safeguard — but it prevents a future LLM call added
                # elsewhere in the graph from silently leaking into this
                # stream unexpectedly.
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name not in (None, "agent"):
                    continue

                # Filter 2: skip chunks that are tool-call deltas rather than
                # visible answer text. When create_react_agent's model
                # decides to call a tool, it streams structured tool-call
                # data (tool_call_chunks / additional_kwargs), often with
                # empty .content — this guards against ever forwarding a
                # partial tool-call fragment as if it were answer text,
                # even though in practice these queries didn't trigger tool
                # calls (single LLM call per turn, confirmed by chunk_count
                # staying consistent with typical single-pass generation).
                has_tool_call_chunk = bool(getattr(chunk, "tool_call_chunks", None))
                content = getattr(chunk, "content", "") if chunk else ""

                if has_tool_call_chunk or not content:
                    skipped_tool_chunks += 1
                    continue

                chunk_count += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                    print(f"Time to first token: {first_token_time:.2f}s")

                payload = json.dumps({"token": content})
                yield f"data: {payload}\n\n"

            elapsed = time.perf_counter() - start
            print(
                f"Stream completed in {elapsed:.2f}s "
                f"(first token at {first_token_time}, "
                f"{chunk_count} content chunks, {skipped_tool_chunks} skipped)"
            )

            done_payload = json.dumps({
                "done": True,
                "elapsed_s": round(elapsed, 2),
                "first_token_s": round(first_token_time, 2) if first_token_time else None,
            })
            yield f"data: {done_payload}\n\n"

        except Exception as exc:
            import traceback
            print("\n--- STREAM FAILED ---")
            traceback.print_exc()
            error_payload = json.dumps({"error": str(exc)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )