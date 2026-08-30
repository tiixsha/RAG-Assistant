"""
src/graph.py — Day 6: full pipeline wired as a LangGraph graph.
retrieval node -> agent/tool node -> generation node
"""

from typing import TypedDict, Annotated
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage


from src.llm import llm
from src.rag import build_vectorstore, get_retriever, load_documents, chunk_documents
from src.agent import calculator, doc_search


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    retrieved_context: str


# --- Build the retriever once at import time ---
# (assumes Qdrant already has the collection populated from Day 4's run;
# set recreate=False so we don't re-embed on every import)
_documents = load_documents()
_chunks = chunk_documents(_documents)
_vectorstore = build_vectorstore(_chunks, recreate=False)
_retriever = get_retriever(_vectorstore, k=3)


@tool
def retrieve_papers(query: str) -> str:
    """Search the speaker-embedding papers (x-vectors, ECAPA-TDNN) for relevant context."""
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant content found in the source papers."
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page')}] {doc.page_content}"
        for doc in docs
    )


# Agent now has three tools: calculator, doc_search (fake demo tool from Day 2),
# and the real retriever tool.
tools = [calculator, doc_search, retrieve_papers]
agent_executor = create_react_agent(llm, tools)


def retrieve_node(state: GraphState) -> GraphState:
    """Pulls the latest human question and does an upfront retrieval pass."""
    last_message = state["messages"][-1]
    query = last_message.content
    docs = _retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"messages": [], "retrieved_context": context}


def agent_node(state: GraphState) -> GraphState:
    """Runs the ReAct agent, grounded in the upfront-retrieved context.
    Only returns the NEW messages the agent produced, so the reducer
    doesn't double-count the original human message."""
    context = state.get("retrieved_context", "")
    grounding = SystemMessage(
        content=(
            "Use the following retrieved context from the source papers "
            "to answer the user's question. If the context doesn't contain "
            "the answer, say so rather than relying on outside knowledge.\n\n"
            f"Context:\n{context}"
        )
    )
    input_messages = [grounding] + state["messages"]
    result = agent_executor.invoke({"messages": input_messages})

    new_messages = result["messages"][len(input_messages):]
    return {"messages": new_messages, "retrieved_context": context}


def generate_node(state: GraphState) -> GraphState:
    """Final pass-through node. The agent's last message is already the answer;
    this node exists so the graph has a distinct generation step for the
    architecture diagram, and as a hook for future post-processing (Day 9+)."""
    return state


# --- Assemble the graph ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("agent", agent_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "agent")
workflow.add_edge("agent", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()


if __name__ == "__main__":

    graph.get_graph().draw_mermaid_png(output_file_path="docs/architecture_diagram.png")
    print("Saved architecture diagram to docs/architecture_diagram.png")

    
    result = graph.invoke({
        "messages": [HumanMessage(content="What is the ECAPA-TDNN architecture?")],
        "retrieved_context": "",
    })
    print("\n--- Final messages ---")
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")