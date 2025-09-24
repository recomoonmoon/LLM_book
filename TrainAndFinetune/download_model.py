from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import os


# 配置国内源和缓存路径（指定缓存路径可以不需要）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = r"./models"

model_name = "Qwen/Qwen3-0.6B"
target_dir = r"./models/Qwen3-0.6B"

snapshot_download(
    repo_id=model_name,
    local_dir=target_dir,
    local_dir_use_symlinks=False
)