from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace

device = "GPU"
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
model_path = "models/ovmodels/phi-3-mini-instruct"

# Create the HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
  model_id=model_path,
  task="text-generation",
  backend="openvino",
  model_kwargs={
    "device": device,
    "ov_config": ov_config,
  },
  pipeline_kwargs={"max_new_tokens": 2048},
)

chat = ChatHuggingFace(llm=llm, verbose=True)
