import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('JeffDing/dangwu_1_8b', cache_dir='/root/model/', revision='master')