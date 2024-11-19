from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  from models.modelOV import llm
elif backend == "cuda" or backend == "cpu":
  raise NotImplementedError("This backend is not supported yet.")
elif backend == "ollama":
  from models.modelOllama import llm
else:
  raise ValueError(f"Unknown backend: {backend}")

import gradio as gr
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("You are a helpful assistant that answers the following question: {messages}")
chain = prompt | llm

def partial_text_processor(partial_text, new_text):
  partial_text += new_text
  return partial_text

def user(message, history):
  return "", history + [[message, ""]]

def bot(history):
  partial_text = ""
  for chunk in chain.stream({"messages": history[-1][0] }):
    partial_text = partial_text_processor(partial_text, chunk)
    history[-1][1] = partial_text
    yield history

def request_cancel():
  chain.cancel()

with gr.Blocks(
  theme=gr.themes.Citrus(),
  css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
  gr.Markdown(f"""<h1><center>Getac Chatbot</center></h1>""")
  chatbot = gr.Chatbot(height=500)
  with gr.Row():
    with gr.Column():
      msg = gr.Textbox(
        label="Chat Message Box",
        placeholder="Chat Message Box",
        show_label=False,
        container=False,
      )
    with gr.Column():
      with gr.Row():
        submit = gr.Button("Submit")
        stop = gr.Button("Stop")
        clear = gr.Button("Clear")

  submit_event = msg.submit(
    fn=user,
    inputs=[msg, chatbot],
    outputs=[msg, chatbot],
    queue=False,
  ).then(
    fn=bot,
    inputs=[
        chatbot,
    ],
    outputs=chatbot,
    queue=True,
  )
  submit_click_event = submit.click(
    fn=user,
    inputs=[msg, chatbot],
    outputs=[msg, chatbot],
    queue=False,
  ).then(
    fn=bot,
    inputs=[
        chatbot,
    ],
    outputs=chatbot,
    queue=True,
  )
  stop.click(
    fn=request_cancel,
    inputs=None,
    outputs=None,
    cancels=[submit_event, submit_click_event],
    queue=False,
  )
  clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
  demo.launch()
