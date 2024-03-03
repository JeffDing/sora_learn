from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch
from modelscope import snapshot_download

model_dir_lcm = snapshot_download("AI-ModelScope/lcm-sdxl",revision = "master")
model_dir_sdxl = snapshot_download("AI-ModelScope/stable-diffusion-xl-base-1.0",revision = "v1.0.9")

unet = UNet2DConditionModel.from_pretrained(model_dir_lcm, torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained(model_dir_sdxl, unet=unet, torch_dtype=torch.float16, variant="fp16")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "Beautiful and cute girl, 16 years old, rain jacket, gradient background, soft colors, soft lighting, cinematic edge lighting, light and dark contrast, anime, art station Seraflur, blind box, super detail, 8k"
image = pipe(prompt, num_inference_steps=4, guidance_scale=8.0).images[0]
image.save("SDXLLCM.png")