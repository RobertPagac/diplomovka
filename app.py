import bentoml
from bentoml.io import JSON
import torch
from diffusers import StableDiffusionXLPipeline
import uuid
import base64
from io import BytesIO
from PIL import Image

@bentoml.service
class SDXLTurbo:
    def __init__(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    @bentoml.api(input=JSON(), output=JSON())
    def txt2img(self, request):
        prompt = request['prompt']
        num_steps = request.get('num_steps', 1)
        guidance_scale = request.get('guidance_scale', 0.0)

        with torch.no_grad():
            result = self.pipe(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            ).images[0]

        return {"image": self.encode_image(result)}

    def encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
