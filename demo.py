from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  from models.modelOV import ov_llm
elif backend == "cuda" or backend == "cpu":
  raise NotImplementedError("This backend is not supported yet.")
else:
  raise ValueError(f"Unknown backend: {backend}")

import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

def chatbot(message, history):
  prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {messages}"
  )

  chain = prompt | ov_llm

  json_parser = SimpleJsonOutputParser()
  response = chain.invoke({"messages": message })
  json_output = json_parser.parse(response)
  return json_output['answer']

demo = gr.ChatInterface(fn=chatbot, type="messages", title="Chatbot")

if __name__ == "__main__":
  demo.launch()
