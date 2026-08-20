from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise SystemExit(
        "GROQ_API_KEY not found. Check that .env exists in this folder "
        "and contains GROQ_API_KEY=your_key_here"
    )

llm = ChatGroq(model="openai/gpt-oss-120b")  # check console.groq.com for current model names if this errors

response = llm.invoke("Say hello in one sentence.")

print("Success — Groq responded:")
print(response.content)