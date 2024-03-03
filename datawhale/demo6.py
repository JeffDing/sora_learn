from diffusers import AutoPipelineForText2Image
from modelscope import snapshot_download
import torch

model_dir=snapshot_download("YorickHe/majicmixRealistic_v6")
lora_dir = snapshot_download("PaperCloud/zju19_dunhuang_style_lora")

pipeline = AutoPipelineForText2Image.from_pretrained(f"{model_dir}/v7", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(lora_dir, weight_name="dunhuang.safetensors")
prompt = "1 girl, close-up, waist shot, black long hair, clean face, dunhuang, Chinese ancient style, clean skin, organza_lace, Dunhuang wind, Art deco, Necklace, jewelry, Bracelet, Earrings, dunhuang_style, see-through_dress, Expressionism, looking towards the camera, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, photorealistic, best quality"
#Negative Prompt = """(nsfw:2), paintings, sketches, (worst quality:2), (low quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, bad hand, tattoo, (username, watermark, signature, time signature, timestamp, artist name, copyright name, copyright),low res, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, strange fingers, bad hand, mole, ((extra legs)), ((extra hands))"""
image = pipeline(prompt).images[0]
image.save("sdlora.png")