import gradio as gr
from model import load_model


def load_interface() -> gr.Interface:
    model = load_model()
    text_to_image_interface = gr.Interface(
        fn=model.generate_image,
        inputs=["text", "text"],
        outputs=["image"],
    )
    return text_to_image_interface


if __name__ == "__main__":
    interface = load_interface()
    interface.launch()
