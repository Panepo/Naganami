from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  raise ValueError(f"openvino backend is not working in this version")

import gradio as gr
from agent import app
from langchain_core.messages import HumanMessage

def gradio_interface(user_input):
    final_state = app.invoke({"messages": [HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": 42}})
    return final_state["messages"][-1].content

iface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text")
iface.launch()
