from typing import TypedDict, Annotated
import asyncio
import operator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.llm import llm, local_llm

from src.rag import (
    build_vectorstore,
    get_retriever,
    load_documents,
    chunk_documents,
    get_embeddings,
    COLLECTION_NAME,
    QDRANT_URL,
)
from src.agent import calculator, doc_search


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    retrieved_context: str


# -------------------------------------------------------------------
# Load documents and build/connect to vector store
# -------------------------------------------------------------------
# Guard against re-embedding on every import (e.g. every uvicorn --reload
# restart). Only run the full ingest-and-upsert path if the collection is
# missing or empty; otherwise just connect to what's already there.
# -------------------------------------------------------------------

_client = QdrantClient(url=QDRANT_URL)

_collection_ready = (
    _client.collection_exists(COLLECTION_NAME)
    and _client.get_collection(COLLECTION_NAME).points_count > 0
)

if _collection_ready:
    print(f"Qdrant collection '{COLLECTION_NAME}' already populated — skipping re-ingestion.")
    _vectorstore = QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )
else:
    print(f"Qdrant collection '{COLLECTION_NAME}' missing or empty — ingesting.")
    _documents = load_documents()
    _chunks = chunk_documents(_documents)
    _vectorstore = build_vectorstore(
        _chunks,
        recreate=False,
    )

_retriever = get_retriever(
    _vectorstore,
    k=3,
)


# -------------------------------------------------------------------
# Retrieval tool
# -------------------------------------------------------------------
@tool
async def retrieve_papers(query: str) -> str:
    """Search the speaker-embedding papers for relevant context."""

    docs = await _retriever.ainvoke(query)

    if not docs:
        return "No relevant content found in the source papers."

    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')} "
        f"p.{doc.metadata.get('page', 'unknown')}] "
        f"{doc.page_content}"
        for doc in docs
    )


# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------

tools = [
    calculator,
    doc_search,
    retrieve_papers,
]

agent_executor = create_react_agent(
    llm,
    tools,
)

# Fallback agent, same tools, bound to the local vLLM model instead of Groq.
# Built eagerly (not lazily) so a Groq outage doesn't also pay the cost of
# constructing this on the failure path.
agent_executor_fallback = create_react_agent(
    local_llm,
    tools,
)


# -------------------------------------------------------------------
# Retrieve node
# -------------------------------------------------------------------

async def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant document chunks for the user's query."""

    last_message = state["messages"][-1]

    query = last_message.content

    docs = await _retriever.ainvoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "messages": [],
        "retrieved_context": context,
    }


# -------------------------------------------------------------------
# Agent node
# -------------------------------------------------------------------

async def agent_node(state: GraphState) -> GraphState:
    """Run the ReAct agent using the retrieved document context.
    Tries Groq first; falls back to the local vLLM model on failure.

    Manual try/except instead of llm.with_fallbacks(), because
    RunnableWithFallbacks doesn't implement bind_tools() and breaks
    create_react_agent (LangGraph #4754). See NOTES.md."""

    context = state.get(
        "retrieved_context",
        "",
    )

    grounding = SystemMessage(
        content=(
            "Use the following retrieved context from the source papers "
            "to answer the user's question. "
            "If the context does not contain the answer, "
            "say so rather than relying on outside knowledge.\n\n"
            f"Context:\n{context}"
        ),
    )

    input_messages = [
        grounding,
        *state["messages"],
    ]

    try:
        result = await agent_executor.ainvoke(
            {
                "messages": input_messages,
            }
        )
    except Exception as e:
        print(f"Groq call failed ({e!r}); falling back to local vLLM model.")
        result = await agent_executor_fallback.ainvoke(
            {
                "messages": input_messages,
            }
        )

    # Keep only messages generated by the agent.
    new_messages = result["messages"][len(input_messages):]

    return {
        "messages": new_messages,
        "retrieved_context": context,
    }

# -------------------------------------------------------------------
# Generate node
# -------------------------------------------------------------------

async def generate_node(state: GraphState) -> GraphState:
    """
    Final generation stage.

    The agent currently performs the actual generation,
    so this node simply passes the state forward.
    """

    return state


# -------------------------------------------------------------------
# Build LangGraph workflow
# -------------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node(
    "retrieve",
    retrieve_node,
)

workflow.add_node(
    "agent",
    agent_node,
)

workflow.add_node(
    "generate",
    generate_node,
)

workflow.set_entry_point("retrieve")

workflow.add_edge(
    "retrieve",
    "agent",
)

workflow.add_edge(
    "agent",
    "generate",
)

workflow.add_edge(
    "generate",
    END,
)

graph = workflow.compile()


# -------------------------------------------------------------------
# Async test
# -------------------------------------------------------------------

async def test_async_graph():
    """Run a simple asynchronous test of the graph."""

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "What is statistics pooling "
                        "in the x-vector system?"
                    )
                )
            ],
            "retrieved_context": "",
        }
    )

    print("\n--- Async Graph Result ---")

    for message in result["messages"]:
        print(
            f"[{message.type}] {message.content}"
        )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":

    # Generate architecture diagram
    graph.get_graph().draw_mermaid_png(
        output_file_path="docs/architecture_diagram.png"
    )

    print(
        "Saved architecture diagram to "
        "docs/architecture_diagram.png"
    )

    # Run async graph test
    asyncio.run(
        test_async_graph()
    )