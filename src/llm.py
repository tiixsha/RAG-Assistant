import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=1.0,
)

def add_user_message(messages: list, content: str) -> list:
    messages.append(HumanMessage(content=content))
    return messages

def add_assistant_message(messages: list, content: str) -> list:
    messages.append(AIMessage(content=content))
    return messages


class QueryAnalysis(BaseModel):
    topic: str = Field(description="The main topic of the user's question")
    is_question: bool = Field(description="Whether the input is a question")
    sentiment: str = Field(description="One of: positive, neutral, negative")

structured_llm = llm.with_structured_output(QueryAnalysis)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer clearly and concisely. Today's date is {date}."),
    ("human", "{user_input}"),
])

chain = prompt | llm