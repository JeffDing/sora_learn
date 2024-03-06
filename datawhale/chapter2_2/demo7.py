from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from PIL import Image
from modelscope import snapshot_download
import cv2
import numpy as np
import torch


model_dir = snapshot_download("AI-ModelScope/stable-diffusion-xl-base-1.0")
controlnet_dir = snapshot_download("AI-ModelScope/controlnet-canny-sdxl-1.0")
VAE_dir = snapshot_download("AI-ModelScope/sdxl-vae-fp16-fix")
original_image = load_image(
    "/root/workspace/canny.jpg"
)

prompt = "sea turtle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = load_image("/root/workspace/canny.jpg")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    controlnet_dir,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(VAE_dir, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_dir,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"controlnet.png")