from transformers import ViTForImageClassification
import torch
from modelscope import snapshot_download
from PIL import Image
import requests
from transformers import ViTImageProcessor

model_dir = snapshot_download('AI-ModelScope/vit-base-patch16-224')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViTForImageClassification.from_pretrained(model_dir)
model.to(device)

url = './000000039769.jpg'
image = Image.open(url)

processor = ViTImageProcessor.from_pretrained(model_dir)
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

print(pixel_values.shape)

with torch.no_grad():
  outputs = model(pixel_values)
logits = outputs.logits
print(logits.shape)

prediction = logits.argmax(-1)
print("Predicted class:", model.config.id2label[prediction.item()])