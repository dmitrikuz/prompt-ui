import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image


class StableDiffusionModel:
    def __init__(self):
        self.pipe = None
        self.loaded = False
        self.model_name = "runwayml/stable-diffusion-v1-5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_pretrained(self):

        if not self.loaded:
            params = {
                "pretrained_model_name_or_path": self.model_name,
            }
            if self.device == "cuda":
                params["torch_dtype"] = torch.float16
                params["use_safetensors"] = True
            
            pipe = StableDiffusionPipeline.from_pretrained(**params)
            self.pipe = pipe.to(self.device)
            self.loaded = True

        return self

    def generate_image(self, prompt: str, negative_prompt: str) -> Image:
        output = self.pipe(prompt=prompt, negative_prompt=negative_prompt)
        image = output.images[0]
        return image

stable_model = StableDiffusionModel()
