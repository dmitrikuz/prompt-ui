import gradio as gr
from model import stable_model


def generate_image(prompt, negative_prompt):
    stable_model.load_pretrained()
    return stable_model.generate_image(prompt, negative_prompt)

def start_gradio():
    interface = gr.Interface(
        fn=generate_image,
        inputs=["text", "text"],
        outputs=["image"]
    )
    interface.launch()


if __name__ == "__main__":
    start_gradio()
    
