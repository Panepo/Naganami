from dotenv import load_dotenv
import os

load_dotenv()
model = os.getenv("MODEL")

if model == "llama3.2":
  modelname = "llama3.2"
elif model == "phi3.5":
  modelname = "phi3.5"
else:
  raise ValueError(f"Unknown model: {model}")

from langchain_ollama import OllamaLLM, ChatOllama

llm = OllamaLLM(model=modelname, verbose=True)
chat = ChatOllama(model=modelname, verbose=True)
