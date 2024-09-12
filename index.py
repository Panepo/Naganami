from langchain_core.prompts import ChatPromptTemplate
from models.modelOV import ov_llm

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
    ("human", "{input}")
  ]
)

# Create LLM Chain
chain = prompt | ov_llm

while 1:
  print("================================================")
  text = input("Please say something: ")
  response = chain.invoke({"input": text })
  print("================================================")
  print(response)
