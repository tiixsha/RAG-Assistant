from src.llm import structured_llm, QueryAnalysis

result = structured_llm.invoke("I'm really frustrated — why does my code keep crashing?")

print(result)
print(type(result))

assert isinstance(result, QueryAnalysis), "Output is not a QueryAnalysis instance!"
print("✅ Structured output check passed")