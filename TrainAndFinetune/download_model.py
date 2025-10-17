from huggingface_hub import snapshot_download
import os
import shutil

def download_safetensors_model():
    model_path = './models/m3e-small'
    # 删除旧目录
    if os.path.exists(model_path):
        print(1111)
        shutil.rmtree(model_path)
        # 使用 huggingface_hub 下载，优先 safetensors print("下载 M3E-small 模型 (safetensors 格式)...")
        snapshot_download( repo_id="moka-ai/m3e-small",
                           local_dir=model_path,
                           local_dir_use_symlinks=False,
                           #ignore_patterns=["*.bin", "*.h5"]
                           ) # 忽略旧格式文件
        print(f"下载完成: {model_path}")
        # 下载模型


download_safetensors_model()