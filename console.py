from dotenv import load_dotenv
import os
import time

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  from models.modelOV import ov_llm
elif backend == "cuda" or backend == "cpu":
  raise NotImplementedError("This backend is not supported yet.")
else:
  raise ValueError(f"Unknown backend: {backend}")

config = {"configurable": {"thread_id": "abc345"}}

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {messages}"
)

# Create LLM Chain
chain = prompt | ov_llm

json_parser = SimpleJsonOutputParser()

def demo():
  while 1:
    print("================================================")
    text = input("Please say something: ")
    start_time = time.process_time()
    response = chain.invoke({"messages": text })
    json_output = json_parser.parse(response)
    print(json_output['answer'])
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"Process time: {elapsed_time} seconds")

if __name__ == "__main__":
  demo()
