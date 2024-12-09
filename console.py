from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "cuda" or backend == "cpu":
  from llm_hf import llm, generation_kwargs
elif backend == "ollama":
  from llm_ollama import llm
else:
  raise ValueError(f"Unknown backend: {backend}")

import time
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

prompt = ChatPromptTemplate.from_messages([
  ("system", DEFAULT_SYSTEM_PROMPT),
  ("human", "{messages}")
])

chain = prompt | llm

def demo():
  while 1:
    print("================================================")
    text = input("Please say something: ")
    start_time = time.time()
    if backend == "cuda" or backend == "cpu":
      response = chain.invoke({"messages": [text] }, **generation_kwargs)
    else:
      response = chain.invoke({"messages": text })
    print(response)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process time: {elapsed_time} seconds")

if __name__ == "__main__":
  demo()
