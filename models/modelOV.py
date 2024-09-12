from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

device = "GPU"
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
model_path = "models/ovmodels/Mistral-7B-Instruct-v0.3-ov-int4"

# Create the HuggingFacePipeline
ov_llm = HuggingFacePipeline.from_model_id(
  model_id=model_path,
  task="text-generation",
  backend="openvino",
  model_kwargs={
    "device": device,
    "ov_config": ov_config,
  },
  pipeline_kwargs={"max_new_tokens": 2048},
)
