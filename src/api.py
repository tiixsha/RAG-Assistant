"""
src/api.py — Async FastAPI backend with caching,
retry logic, and rate limiting.
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
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from src.graph import graph


# -------------------------------------------------------------------
# LLM CACHE
# -------------------------------------------------------------------

# Cache LLM responses in memory.
#
# Identical LLM requests can be served from the cache instead
# of calling the LLM again.
#
# Note:
# The cache is cleared whenever the FastAPI process restarts.
# -------------------------------------------------------------------

set_llm_cache(
    InMemoryCache()
)


# -------------------------------------------------------------------
# RATE LIMITING
# -------------------------------------------------------------------

# Maximum number of requests allowed within the time window.
RATE_LIMIT = 10

# Rate-limit window in seconds.
RATE_WINDOW = 60

# Store timestamps of recent requests.
REQUEST_TIMESTAMPS = deque()


def check_rate_limit():
    """
    Check whether the client is within the rate limit.

    Current policy:
        10 requests per 60 seconds.

    This is a simple in-memory limiter suitable for the
    current project/demo.
    """

    now = time.time()

    # Remove timestamps older than the current window.
    while REQUEST_TIMESTAMPS:

        oldest_request = REQUEST_TIMESTAMPS[0]

        if now - oldest_request > RATE_WINDOW:
            REQUEST_TIMESTAMPS.popleft()

        else:
            break

    # Check whether the limit has been reached.
    if len(REQUEST_TIMESTAMPS) >= RATE_LIMIT:

        raise HTTPException(
            status_code=429,
            detail=(
                "Rate limit exceeded. "
                "Please try again later."
            ),
        )

    # Record the current request.
    REQUEST_TIMESTAMPS.append(now)


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="Speaker Embedding AI Assistant",
    description=(
        "Async LangGraph RAG backend "
        "with caching and reliability features"
    ),
    version="1.0.0",
)


# -------------------------------------------------------------------
# Request / Response models
# -------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    trace: list[Any]
    retrieved_context: str


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
    # Start request timer
    # ---------------------------------------------------------------

    start = time.perf_counter()

    # ---------------------------------------------------------------
    # Retry configuration
    # ---------------------------------------------------------------

    max_retries = 3

    # ---------------------------------------------------------------
    # Execute graph with retry logic
    # ---------------------------------------------------------------

    result = None

    for attempt in range(max_retries):

        try:

            print(
                f"Processing request "
                f"(attempt {attempt + 1}/{max_retries})"
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=query
                        )
                    ],
                    "retrieved_context": "",
                }
            )

            # Graph succeeded.
            break

        except Exception as exc:

            print(
                f"Attempt {attempt + 1}/"
                f"{max_retries} failed: {exc}"
            )

            # If this was the final attempt,
            # propagate the error.
            if attempt == max_retries - 1:

                import traceback

                print(
                    "\n--- REQUEST FAILED ---"
                )

                traceback.print_exc()

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "The AI service failed after "
                        "multiple attempts."
                    ),
                )

            # Wait before retrying.
            #
            # Attempt 1 -> 1 second
            # Attempt 2 -> 2 seconds
            await asyncio.sleep(
                attempt + 1
            )

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

    messages = result.get(
        "messages",
        [],
    )

    # ---------------------------------------------------------------
    # Extract final AI answer
    # ---------------------------------------------------------------

    answer = ""

    for message in reversed(messages):

        if getattr(
            message,
            "type",
            None,
        ) == "ai":

            content = message.content

            if isinstance(
                content,
                str,
            ):

                answer = content
                break

    if not answer:

        answer = "No answer was generated."

    # ---------------------------------------------------------------
    # Serialize message trace
    # ---------------------------------------------------------------

    trace = []

    for message in messages:

        # Pydantic v2 / LangChain objects
        if hasattr(
            message,
            "model_dump",
        ):

            trace.append(
                message.model_dump()
            )

        # Pydantic v1 compatibility
        elif hasattr(
            message,
            "dict",
        ):

            trace.append(
                message.dict()
            )

        # Already a dictionary
        elif isinstance(
            message,
            dict,
        ):

            trace.append(
                message
            )

        # Fallback
        else:

            trace.append(
                {
                    "type": getattr(
                        message,
                        "type",
                        "unknown",
                    ),
                    "content": getattr(
                        message,
                        "content",
                        "",
                    ),
                }
            )

    # ---------------------------------------------------------------
    # Retrieved context
    # ---------------------------------------------------------------

    retrieved_context = result.get(
        "retrieved_context",
        "",
    )

    # ---------------------------------------------------------------
    # Request timing
    # ---------------------------------------------------------------

    elapsed = (
        time.perf_counter() - start
    )

    print(
        f"Request completed in "
        f"{elapsed:.2f}s"
    )

    # ---------------------------------------------------------------
    # Return response
    # ---------------------------------------------------------------

    return ChatResponse(
        answer=answer,
        trace=trace,
        retrieved_context=retrieved_context,
    )