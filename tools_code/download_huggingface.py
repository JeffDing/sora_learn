import os

# 下载模型
os.system('huggingface-cli download --resume-download --local-dir-use-symlinks False YaohuiW/LaVie/ --local-dir /root/temp/LaVie/pretrained_models')
os.system('huggingface-cli download --resume-download --local-dir-use-symlinks False CompVis/stable-diffusion-v1-4 --local-dir /root/temp/LaVie/pretrained_models/stable-diffusion-v1-4')
os.system('huggingface-cli download --resume-download --local-dir-use-symlinks False stabilityai/stable-diffusion-x4-upscaler --local-dir /root/temp/LaVie/pretrained_models/stable-diffusion-x4-upscaler')