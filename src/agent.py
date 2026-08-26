from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from src.llm import llm

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '12 * (4 + 3)'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def doc_search(query: str) -> str:
    """Search internal documentation for a topic and return a short summary."""
    fake_docs = {
        "langgraph": "LangGraph is a library for building stateful, multi-step agent workflows as graphs.",
        "groq": "Groq provides fast LLM inference via custom LPU hardware, with a free tier and OpenAI-compatible API.",
        "vllm": "vLLM is an inference engine using PagedAttention for efficient KV cache management.",
    }
    for key, summary in fake_docs.items():
        if key in query.lower():
            return summary
    return "No matching documentation found."

tools = [calculator, doc_search]
agent = create_react_agent(llm, tools)