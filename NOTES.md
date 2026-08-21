## LLM Integration + Prompt Engineering
- Studied Claude Learning Path (API basics, messages) + Prompt Engineering Guide (system prompts, temp/top_p, few-shot). Concept translated cleanly to LangChain — SystemMessage/HumanMessage/AIMessage map directly onto Claude's message roles.
- Wired up ChatGroq (model: openai/gpt-oss-120b — note: llama-3.3-70b-versatile is deprecated as of Aug 2026), confirmed llm.invoke() returns text via sanity_check.py.
- Built add_user_message/add_assistant_message helpers around LangChain message objects.
- Built system prompt with ChatPromptTemplate instead of string concat; chained with `prompt | llm`.
- Compared temperature 0.2 / 0.7 / 1.2 on the same prompt — lower temp gave safe/repetitive phrasing, higher temp gave more varied wording, as expected.
- Added structured output via `llm.with_structured_output(QueryAnalysis)` — confirmed with isinstance() that output is a typed Pydantic object, not raw text/dict.
- Gotcha hit: `from itertools import chain` shadowed the imported `chain` from src.llm — reminder to watch for stdlib name collisions.