import torch
from modelscope import snapshot_download
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

device = "cuda"
num_images_per_prompt = 1

stable_cascade_prior = snapshot_download("AI-ModelScope/stable-cascade-prior")
stable_cascade = snapshot_download("AI-ModelScope/stable-cascade")

prior = StableCascadePriorPipeline.from_pretrained(stable_cascade_prior, torch_dtype=torch.bfloat16,).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained(stable_cascade,  torch_dtype=torch.float16).to(device)

prompt = "Beautiful and cute girl, 16 years old, rain jacket, gradient background, soft colors, soft lighting, cinematic edge lighting, light and dark contrast, anime, art station Seraflur, blind box, super detail, 8k"
negative_prompt = ""

prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
    num_inference_steps=20
)
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.half(),
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images

for i, img in enumerate(decoder_output):
    img.save(f"stablecascade_{i+1}.png")
#Now decoder_output is a list with your PIL images