import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

import gradio as gr


class StableDiffusionModel:
    def __init__(self):
        self.pipe = None
        self.loaded = False
        self.model_name = "runwayml/stable-diffusion-v1-5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_pretrained(self):
        params = {
            "pretrained_model_name_or_path": self.model_name,
        }
        if self.device == "cuda":
            params["torch_dtype"] = torch.float16
            params["use_safetensors"] = True

        if not self.loaded:
            pipe = StableDiffusionPipeline.from_pretrained(**params)
            pipe = pipe.to(self.device)

            self.pipe = pipe
            self.loaded = True

    def generate_image(self, prompt: str, negative_prompt: str) -> Image:
        output = self.pipe(prompt=prompt, negative_prompt=negative_prompt)
        image = output.images[0]
        return image


def load_model() -> StableDiffusionModel:
    model = StableDiffusionModel()
    model.load_pretrained()
    return model


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
