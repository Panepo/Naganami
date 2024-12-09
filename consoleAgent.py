from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

import time
from agent import app
from langchain_core.messages import HumanMessage

def demo():
  while 1:
    print("================================================")
    text = input("Please say something: ")
    start_time = time.time()
    response = app.invoke(
      {"messages": [HumanMessage(content=text)]},
      config={"configurable": {"thread_id": 42}}
    )
    print(response["messages"][-1].content)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process time: {elapsed_time} seconds")

if __name__ == "__main__":
  demo()
