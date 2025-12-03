!pip install gradio diffusers transformers accelerate safetensors torch --quiet

import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image, ImageChops, ImageFilter, ImageOps
import io, base64, numpy as np

# --- model ---
model_id = "dream-textures/texture-diffusion"
pipe_txt = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe_img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")


# --- utils ---
def prepare_image(img):
    if img is None:
        return None
    if isinstance(img, dict):
        if "image" in img:
            b64_data = img["image"].split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))
        elif "composite" in img:
            img = img["composite"]
        else:
            return None
    if img is None:
        return None
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    return img.resize((512, 512))


def make_tile_preview(img, size=3):
    w, h = img.size
    canvas = Image.new("RGB", (w * size, h * size))
    for i in range(size):
        for j in range(size):
            canvas.paste(img, (i * w, j * h))
    return canvas


def combine_inputs(draw_img):
    if not draw_img or draw_img == {}:
        return None
    return prepare_image(draw_img)


def make_seamless(img):
    if img is None:
        return None
    w, h = img.size
    offset_img = ImageChops.offset(img, w // 2, h // 2)
    img_torch = pipe_img(
        prompt="seamless texture, smooth transition, no borders",
        image=offset_img,
        strength=0.4,
        num_inference_steps=20,
        guidance_scale=7.0
    ).images[0]
    return ImageChops.offset(img_torch, -w // 2, -h // 2)

def should_invert_bump(prompt):
  p = prompt.lower()
  return any(word in p for word in ["brick", "bricks", "brick wall"])

# bump map
def make_bump_map(img, prompt):
    base_gray = ImageOps.grayscale(img)
    base_gray = base_gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    base_gray = base_gray.filter(ImageFilter.EDGE_ENHANCE)
    base_gray = base_gray.resize((512, 512))

    if "brick" in prompt.lower():
      base_gray = ImageOps.invert(base_gray)

    # --- STEP 2: AI enhancement (requires RGB!) ---
    bump_prompt = (
        "high quality bump map, height map, displacement map, "
        "white = high, black = low, keep exact texture geometry, "
        "do not change structure, enhance depth only, no new shapes"
    )

    rgb_input = base_gray.convert("RGB")   # <-- FIX

    enhanced = pipe_img(
        prompt=bump_prompt,
        image=rgb_input,
        strength=0.15,
        num_inference_steps=25,
        guidance_scale=7.0
    ).images[0]

    # --- STEP 3: force grayscale output ---
    enhanced = ImageOps.grayscale(enhanced)
    enhanced = ImageOps.autocontrast(enhanced)

    return enhanced



# normal map
def make_normal_map(bump, intensity=2.0):
    bump_np = np.array(bump, dtype=np.float32) / 255.0
    sobel_x = np.array([[ -1, 0, 1 ],
                        [ -2, 0, 2 ],
                        [ -1, 0, 1 ]], dtype=np.float32)
    sobel_y = np.array([[ -1, -2, -1 ],
                        [  0,  0,  0 ],
                        [  1,  2,  1 ]], dtype=np.float32)

    gx = np.zeros_like(bump_np)
    gy = np.zeros_like(bump_np)

    for i in range(1, bump_np.shape[0]-1):
        for j in range(1, bump_np.shape[1]-1):
            region = bump_np[i-1:i+2, j-1:j+2]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    nx = -gx * intensity
    ny = -gy * intensity
    nz = np.ones_like(bump_np)

    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= length
    ny /= length
    nz /= length

    normal = np.stack([(nx + 1) / 2, (ny + 1) / 2, (nz + 1) / 2], axis=-1)
    normal_img = Image.fromarray((normal * 255).astype(np.uint8))
    return normal_img

def generate_texture(prompt, draw_img, strength, steps, guidance, make_seam):
    generator = torch.Generator("cuda").manual_seed(123)
    init_image = combine_inputs(draw_img)

    seamless_hint = ", seamless, tileable, repeating pattern, no visible edges"
    full_prompt = (prompt if prompt else "texture") + seamless_hint

    if init_image is None:
        image = pipe_txt(
            prompt=full_prompt,
            height=512, width=512,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
    else:
        image = pipe_img(
            prompt=full_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]

    if make_seam:
        image = make_seamless(image)

    bump_map = make_bump_map(image, full_prompt)
    normal_map = make_normal_map(bump_map)
    preview = make_tile_preview(image, size=3)

    return image, bump_map, normal_map, preview

# UI
with gr.Blocks() as demo:
    gr.Markdown("## Tileable Texture Generator (with Bump + Normal Maps)")

    with gr.Row():
        prompt = gr.Textbox(label="Enter your texture prompt", value="clear grass texture")

    with gr.Row():
        draw = gr.ImageEditor(label="Draw or edit (optional)", type="pil")

    with gr.Row():
        strength = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Image Strength")
        steps = gr.Slider(10, 50, value=30, step=1, label="Denoising Steps")
        guidance = gr.Slider(1, 12, value=7.5, step=0.1, label="Guidance Scale")

    make_seam = gr.Checkbox(label="Make Seamless", value=True)

    generate_btn = gr.Button("Generate")

    with gr.Row():
        output = gr.Image(label="Color Texture")
        bump = gr.Image(label="Bump Map")
        normal = gr.Image(label="Normal Map")
        preview = gr.Image(label="Tile Preview (3x3)")

    generate_btn.click(
        fn=generate_texture,
        inputs=[prompt, draw, strength, steps, guidance, make_seam],
        outputs=[output, bump, normal, preview]
    )

demo.launch(debug=True)
