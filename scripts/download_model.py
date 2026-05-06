from huggingface_hub import snapshot_download
import os

model_dir = os.path.expanduser("~/models/Llama-3.2-3B-Instruct")

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)