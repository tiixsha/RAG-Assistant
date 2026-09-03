"""
Streamlit UI 

This calls the LangGraph pipeline IN-PROCESS via
graph.invoke(), not over HTTP. The switch to a real HTTP client happens on
Day 8 once the FastAPI wrapper exists 

Run with:
    streamlit run ui/streamlit_app.py
"""

import sys
import os
import time

# Same sys.path boilerplate noted as a Day 2 gotcha — keep it at the top of
# every entrypoint that imports from src/.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.graph import graph  # already-compiled module-level graph object


def to_lc_messages(messages):
    """Convert the simple {'role', 'content'} dicts in session_state into
    LangChain message objects, matching whatever GraphState expects as
    input (adjust if your graph.py takes raw dicts instead)."""
    lc_messages = []
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages


st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------
# Nothing to build here — importing src.graph already builds the retriever
# and vectorstore at module level (with recreate=False, so it connects to
# the existing Day 4 Qdrant collection rather than re-embedding). Python
# only runs that import once per process, so this cost is paid once no
# matter how many times Streamlit reruns the script.
#
# st.cache_resource still earns its keep here: it gives you the loading
# spinner on first load and guarantees every session reuses the same graph
# object instead of anything being rebuilt per-interaction.
@st.cache_resource(show_spinner="Connecting to pipeline (Qdrant, retriever, agent)...")
def get_pipeline():
    return graph


pipeline = get_pipeline()


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
        "in-process call — HTTP wiring lands Day 8"
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
                # In-process call — no requests.post, no FastAPI yet.
                # Full history goes in each time (the graph itself is
                # stateless per invoke, no checkpointing yet); retrieve_node
                # reads state["messages"][-1] as the query, so the newest
                # user turn always ends up last in this list.
                result = pipeline.invoke({
                    "messages": to_lc_messages(st.session_state.messages),
                    "retrieved_context": "",
                })
                elapsed = time.time() - start

                final_messages = result["messages"]
                answer = final_messages[-1].content if final_messages else "(no response)"

                placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Stash whatever LangGraph exposes about intermediate steps
                # for the panel on the right. Adjust the key(s) to match
                # whatever your retrieve_node / agent_node actually put on
                # GraphState (retrieved_context, tool call history, etc.)
                st.session_state.last_trace = {
                    "elapsed_s": round(elapsed, 2),
                    "retrieved_context": result.get("retrieved_context"),
                    "raw_messages": final_messages,
                }

            except Exception as e:
                placeholder.markdown(f"⚠️ Error: `{e}`")
                st.session_state.last_trace = None

with trace_col:
    st.subheader("Intermediate steps")

    trace = st.session_state.last_trace
    if trace is None:
        st.info("Ask a question to see retrieved chunks and tool calls here.")
    else:
        st.metric("Latency", f"{trace['elapsed_s']}s")

        with st.expander("Retrieved context", expanded=True):
            ctx = trace.get("retrieved_context")
            if ctx:
                st.code(ctx if isinstance(ctx, str) else str(ctx), language=None)
            else:
                st.write("No retrieved context on this turn.")

        with st.expander("Full message trace (tool calls, reasoning steps)"):
            for m in trace["raw_messages"]:
                role = getattr(m, "type", m.__class__.__name__)
                content = getattr(m, "content", "")
                tool_calls = getattr(m, "tool_calls", None)
                st.markdown(f"**{role}**")
                if content:
                    st.text(content)
                if tool_calls:
                    st.json(tool_calls)
                st.divider()


# ---------------------------------------------------------------------------
# Sidebar: session controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Session")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()

    st.caption(
        "This UI talks to the LangGraph pipeline in-process. "
        "On Day 8 this becomes a `requests.post(...)` call against "
        "a FastAPI backend instead."
    )