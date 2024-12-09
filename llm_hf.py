from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")
model = os.getenv("MODEL")

from model_download import llama32_dir, llama32_ins_dir, phi35_dir

if model == "llama3.2":
  model_dir = llama32_dir
elif model == "llama3.2ins":
  model_dir = llama32_ins_dir
elif model == "phi3.5":
  model_dir = phi35_dir
else:
  raise ValueError(f"Unknown model: {model}")

if backend == "cuda":
  device = "cuda"
elif backend == "cpu":
  device = "cpu"
else:
  raise ValueError(f"Unknown backend: {backend}")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
  model_dir,
  device_map=device,
  torch_dtype="auto",
  quantization_config=quantization_config,
  trust_remote_code=True,
  attn_implementation='eager'
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

generation_kwargs = {
  "max_new_tokens": 2048,
  "return_full_text": False,
  "temperature": 0.1,
  "top_p": 1.0,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "do_sample": False,
}
