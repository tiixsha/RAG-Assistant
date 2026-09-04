"""
Streamlit UI — talks to the FastAPI backend over HTTP, and now forwards
conversation history on every request so the backend can resolve pronoun
references ("what's its EER?") to earlier turns.

Prerequisite: FastAPI must be running separately:
    myenv\\Scripts\\python.exe -m uvicorn src.api:app --reload --port 8000

Run this with:
    myenv\\Scripts\\streamlit.exe run ui/streamlit_app.py

History handling: /chat's ChatRequest now accepts an optional `history`
field (list of {role, content}), forwarded from st.session_state.messages
(everything before the current turn). api.py converts this into
HumanMessage/AIMessage objects and prepends it to the current query before
calling graph.ainvoke(). Confirmed working via the pronoun-only follow-up
test that originally exposed the missing-history bug.
"""

import os
import time

import streamlit as st
import requests

st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="wide")

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

if "last_trace" not in st.session_state:
    st.session_state.last_trace = None  # intermediate steps from the last run


# ---------------------------------------------------------------------------
# Layout: chat on the left, intermediate-steps panel on the right
# ---------------------------------------------------------------------------
chat_col, trace_col = st.columns([2, 1])

with chat_col:
    st.title("🔎 RAG Assistant")
    st.caption(
        "Groq (primary) · LangGraph agent · Qdrant retrieval · "
        f"HTTP call to FastAPI backend at {API_BASE_URL}"
    )

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about x-vectors, ECAPA-TDNN, or anything in scope...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")

            start = time.time()
            try:
                # HTTP call — replaces the Day 7 in-process pipeline.invoke().
                # Full prior history is now sent alongside the query so the
                # backend can resolve follow-up references correctly.
                # Send everything before this turn as history — the
                # message we just appended for user_input is excluded here
                # since the backend takes it separately as `query`.
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                response = requests.post(
                    CHAT_ENDPOINT,
                    json={"query": user_input, "history": history_payload},
                    timeout=60,
                )
                elapsed = time.time() - start

                if response.status_code == 429:
                    answer = "⏳ Rate limit exceeded. Please wait a moment and try again."
                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_trace = None

                elif response.status_code == 400:
                    detail = response.json().get("detail", "Bad request.")
                    answer = f"⚠️ {detail}"
                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_trace = None

                elif response.status_code != 200:
                    detail = response.json().get("detail", response.text)
                    answer = f"⚠️ Backend error ({response.status_code}): {detail}"
                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_trace = None

                else:
                    data = response.json()
                    answer = data.get("answer", "(no response)")

                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # /chat already serializes messages to plain dicts
                    # (model_dump()/.dict()), so the trace panel below reads
                    # dict keys, not LangChain message attributes.
                    st.session_state.last_trace = {
                        "elapsed_s": round(elapsed, 2),
                        "retrieved_context": data.get("retrieved_context"),
                        "raw_messages": data.get("trace", []),
                    }

            except requests.exceptions.ConnectionError:
                answer = (
                    f"⚠️ Couldn't reach the backend at {API_BASE_URL}. "
                    "Is `uvicorn src.api:app` running?"
                )
                placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_trace = None

            except requests.exceptions.Timeout:
                answer = "⚠️ Request timed out waiting for the backend."
                placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_trace = None

            except Exception as e:
                placeholder.markdown(f"⚠️ Error: `{e}`")
                st.session_state.last_trace = None

with trace_col:
    st.subheader("Intermediate steps")

    trace = st.session_state.last_trace
    if trace is None:
        st.info("Ask a question to see retrieved chunks and tool calls here.")
    else:
        st.metric("Round-trip latency", f"{trace['elapsed_s']}s")
        st.caption("Includes HTTP overhead, not just graph execution time.")

        with st.expander("Retrieved context", expanded=True):
            ctx = trace.get("retrieved_context")
            if ctx:
                st.code(ctx if isinstance(ctx, str) else str(ctx), language=None)
            else:
                st.write("No retrieved context on this turn.")

        with st.expander("Full message trace (tool calls, reasoning steps)"):
            for m in trace["raw_messages"]:
                # These are plain dicts now (serialized server-side), not
                # LangChain message objects — use .get(), not getattr().
                role = m.get("type", "unknown") if isinstance(m, dict) else str(m)
                content = m.get("content", "") if isinstance(m, dict) else ""
                tool_calls = m.get("tool_calls") if isinstance(m, dict) else None

                st.markdown(f"**{role}**")
                if content:
                    st.text(content)
                if tool_calls:
                    st.json(tool_calls)
                st.divider()


# ---------------------------------------------------------------------------
# Sidebar: session controls + backend health
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Session")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()

    st.divider()
    st.subheader("Backend status")
    try:
        health = requests.get(HEALTH_ENDPOINT, timeout=3)
        if health.status_code == 200:
            st.success(f"Connected — {API_BASE_URL}")
        else:
            st.warning(f"Backend responded with {health.status_code}")
    except requests.exceptions.RequestException:
        st.error(f"Can't reach {API_BASE_URL}")

    st.caption(
        "This UI calls the FastAPI backend over HTTP, forwarding the full "
        "conversation history alongside each query so follow-up questions "
        "resolve correctly."
    )