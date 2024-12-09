from pathlib import Path

model_dir = Path("models")

llama32_dir = model_dir / "Llama-3.2-3B"
llama32_ins_dir = model_dir / "Llama-3.2-3B-Instruct"
phi35_dir = model_dir / "Phi-3.5-mini-instruct"

def git_clone(repo, path):
  import subprocess  # nosec - disable B404:import-subprocess check
  subprocess.run(["git", "clone", repo, path], check=True)

def ollama_pull(model):
  import subprocess  # nosec - disable B404:import-subprocess check
  subprocess.run(["ollama", "pull", model], check=True)

def download_model():
  if not model_dir.exists():
    model_dir.mkdir()

  #if not llama32_dir.exists():
  #  git_clone("https://huggingface.co/meta-llama/Llama-3.2-3B", llama32_dir)
  #  ollama_pull("llama3.2:3b")

  #if not llama32_ins_dir.exists():
  #  git_clone("https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct", llama32_ins_dir)

  if not phi35_dir.exists():
    git_clone("https://huggingface.co/microsoft/Phi-3.5-mini-instruct", phi35_dir)
    ollama_pull("phi3.5")

if __name__ == "__main__":
  download_model()
