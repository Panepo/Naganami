from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "cuda" or backend == "cpu":
  raise NotImplementedError("HF not implemented")
elif backend == "ollama":
  from llm_ollama import chat
else:
  raise ValueError(f"Unknown backend: {backend}")

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tools.weather import get_weather
from tools.wiki import wikipedia
from tools.device import device_query

tools = [wikipedia, get_weather, device_query]
tool_node = ToolNode(tools)
formatted_tools = [tool for tool in tools]
model_with_tools = chat.bind_tools(formatted_tools)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
  messages = state['messages']
  last_message = messages[-1]
  # If the LLM makes a tool call, then we route to the "tools" node
  if isinstance(last_message, AIMessage) and last_message.tool_calls:
      return "tools"
  # Otherwise, we stop (reply to the user)
  return END


# Define the function that calls the model
def call_model(state: MessagesState):
  messages = state['messages']
  response = model_with_tools.invoke(messages)
  # We return a list, because this will get added to the existing list
  return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)
