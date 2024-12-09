from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "cuda" or backend == "cpu":
  from llm_hf import llm, generation_kwargs
elif backend == "ollama":
  from llm_ollama import llm
else:
  raise ValueError(f"Unknown backend: {backend}")

import gradio as gr
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

prompt = ChatPromptTemplate.from_messages([
  ("system", DEFAULT_SYSTEM_PROMPT),
  ("human", "{messages}")
])

chain = prompt | llm

def partial_text_processor(partial_text, new_text):
  partial_text += new_text
  return partial_text

def user(message, history):
  return "", history + [[message, ""]]

def bot(history):
  print(f"String length: {len(history[-1][0])}")
  calTime = True

  partial_text = ""
  if backend == "cuda" or backend == "cpu":
    for chunk in chain.stream({"messages": history[-1][0] }, **generation_kwargs):
      partial_text = partial_text_processor(partial_text, chunk)
      history[-1][1] = partial_text
      yield history
  else:
    for chunk in chain.stream({"messages": history[-1][0] }):
      partial_text = partial_text_processor(partial_text, chunk)
      history[-1][1] = partial_text
      yield history

def request_cancel():
  raise NotImplementedError("Cancel not supported")

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
