import os
import torch
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import libs.autoencoder
from libs.uvit import UViT
import einops
from torchvision.utils import save_image
from PIL import Image
import os

from modelscope.hub.file_download import model_file_download

image_size = "256" #@param [256, 512]
image_size = int(image_size)

if image_size == 256:
    model_file_download(model_id='thu-ml/imagenet256_uvit_huge',file_path='imagenet256_uvit_huge.pth', cache_dir='/root/temp/workspace')
    os.system("mv /root/temp/workspace/thu-ml/imagenet256_uvit_huge/imagenet256_uvit_huge.pth /root/temp/workspace/U-ViT/imagenet256_uvit_huge.pth")
else:
    model_file_download(model_id='thu-ml/imagenet512_uvit_huge',file_path='imagenet512_uvit_huge.pth', cache_dir='/root/temp/workspace')
    os.system("mv /root/temp/workspace/thu-ml/imagenet512_uvit_huge/imagenet512_uvit_huge.pth /root/temp/workspace/U-ViT/imagenet512_uvit_huge.pth")
 
z_size = image_size // 8
patch_size = 2 if image_size == 256 else 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

nnet = UViT(img_size=z_size,
       patch_size=patch_size,
       in_chans=4,
       embed_dim=1152,
       depth=28,
       num_heads=16,
       num_classes=1001,
       conv=False)

nnet.to(device)
nnet.load_state_dict(torch.load(f'/root/temp/workspace/U-ViT/imagenet{image_size}_uvit_huge.pth', map_location='cpu'))
nnet.eval()

model_file_download(model_id='AI-ModelScope/autoencoder_kl_ema',file_path='autoencoder_kl_ema.pth', cache_dir='/root/temp/workspace')
os.system("mv /root/temp/workspace/AI-ModelScope/autoencoder_kl_ema/autoencoder_kl_ema.pth /root/temp/workspace/U-ViT/autoencoder_kl_ema.pth")
autoencoder = libs.autoencoder.get_model('/root/temp/workspace/U-ViT/autoencoder_kl_ema.pth')
autoencoder.to(device)

seed = 4321 #@param {type:"number"}
steps = 25 #@param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 3 #@param {type:"slider", min:0, max:10, step:0.1}
class_labels = 207, 360, 387, 974, 88, 979, 417, 279 #@param {type:"raw"}
samples_per_row = 4 #@param {type:"number"}
torch.manual_seed(seed)

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


_betas = stable_diffusion_beta_schedule()  # set the noise schedule
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())


y = torch.tensor(class_labels, device=device)
y = einops.repeat(y, 'B -> (B N)', N=samples_per_row)

def model_fn(x, t_continuous):
    t = t_continuous * len(_betas)
    _cond = nnet(x, t, y=y)
    _uncond = nnet(x, t, y=torch.tensor([1000] * x.size(0), device=device))
    return _cond + cfg_scale * (_cond - _uncond)  # classifier free guidance


z_init = torch.randn(len(y), 4, z_size, z_size, device=device)
dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

with torch.no_grad():
  with torch.cuda.amp.autocast():  # inference with mixed precision
    z = dpm_solver.sample(z_init, steps=steps, eps=1. / len(_betas), T=1.)
    samples = autoencoder.decode(z)
samples = 0.5 * (samples + 1.)
samples.clamp_(0., 1.)
save_image(samples, "sample.png", nrow=samples_per_row * 2, padding=0)