import time
from agent import app
from langchain_core.messages import HumanMessage

start_time = time.process_time()
response = app.invoke(
  {"messages": [HumanMessage(content="what is the weather in sf")]},
  config={"configurable": {"thread_id": 42}}
)
print(response)
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"Process time: {elapsed_time} seconds")


