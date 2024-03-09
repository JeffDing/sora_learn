from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="YaohuiW/LaVie",
  local_dir="/root/temp/LaVie/pretrained_models/",
  max_workers=8
)