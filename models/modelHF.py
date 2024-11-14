from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "cuda":
  device = "cuda"
elif backend == "cpu":
  device = "cpu"
else:
  raise ValueError(f"Unknown backend: {backend}")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline

model_path = "models/hfmodels/Phi-3-mini-4k-instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, quantization_config=quantization_config, torch_dtype="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
llm = HuggingFacePipeline(pipeline=pipe)
