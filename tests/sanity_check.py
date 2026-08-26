import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import agent

response = agent.invoke({"messages": [("user", "What is 15 * 7?")]})
print(response["messages"][-1].content)

response2 = agent.invoke({"messages": [("user", "What is the capital of France?")]})
for msg in response2["messages"]:
    print(msg)

response3 = agent.invoke({"messages": [("user", "Search the docs for 'groq', then multiply 8 by the number of words in that summary.")]})
for msg in response3["messages"]:
    print(msg)