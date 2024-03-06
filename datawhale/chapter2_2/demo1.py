from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                model='AI-ModelScope/stable-diffusion-xl-base-1.0',
                use_safetensors=True,
                model_revision='v1.0.0')

prompt = "Beautiful and cute girl, 16 years old, rain jacket, gradient background, soft colors, soft lighting, cinematic edge lighting, light and dark contrast, anime, art station Seraflur, blind box, super detail, 8k"
output = pipe({'text': prompt})
cv2.imwrite('SDXL.png', output['output_imgs'][0])