from langchain_ollama import OllamaLLM, ChatOllama

llm = OllamaLLM(model="llama3.2", verbose=True)
chat = ChatOllama(model="llama3.2", verbose=True)
