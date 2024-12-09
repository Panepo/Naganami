from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  raise ValueError(f"openvino backend is not working in this version")

import gradio as gr
from agent import app
from langchain_core.messages import HumanMessage

def partial_text_processor(partial_text, new_text):
  partial_text += new_text
  return partial_text

def user(message, history):
  return "", history + [[message, ""]]

async def bot(history):
  response = app.invoke({"messages": [HumanMessage(content=history[-1][0])]}, config={"configurable": {"thread_id": 42}})
  partial_text = ""
  partial_text = partial_text_processor(partial_text, response["messages"][-1].content)
  history[-1][1] = partial_text
  yield history

  #partial_text = ""
  #async for event in app.astream_events({"messages": [HumanMessage(content=history[-1][0])]}, config={"configurable": {"thread_id": 42}}, version="v1"):
  #  if "message" in event.keys():
  #    partial_text = partial_text_processor(partial_text, event.data["message"].content)
  #    history[-1][1] = partial_text
  #    yield history

def request_cancel():
  app.cancel()

with gr.Blocks(
  theme=gr.themes.Citrus(),
  css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
  gr.Markdown(f"""<h1><center>Getac Agent</center></h1>""")
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
