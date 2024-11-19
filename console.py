from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  from models.modelOV import llm
elif backend == "cuda" or backend == "cpu":
  from models.modelHF import llm
elif backend == "ollama":
  from models.modelOllama import llm
else:
  raise ValueError(f"Unknown backend: {backend}")

import time
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

prompt = PromptTemplate.from_template("You are a helpful assistant that answers the following question: {messages}")

# Create LLM Chain
chain = prompt | llm

def demo():
  while 1:
    print("================================================")
    text = input("Please say something: ")
    start_time = time.process_time()
    response = chain.invoke({"messages": text })
    print(response)
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"Process time: {elapsed_time} seconds")

if __name__ == "__main__":
  demo()
