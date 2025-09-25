
## 模型的下载与推理

在国内直接从 HuggingFace 下载模型时，经常会遇到网络不稳定、断点下载失败等问题。为了避免这些情况，可以 **配置国内镜像源** 并 **指定缓存路径** 来保证下载过程更稳定、更可控。

---

### 📌 配置国内源与缓存路径

```python
import os
from huggingface_hub import snapshot_download

# 配置国内镜像（hf-mirror），提升下载成功率
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置本地缓存目录（可选，不设置则默认 ~/.cache/huggingface/）
os.environ["HF_HOME"] = r"./models"

# 模型名称
model_name = "Qwen/Qwen3-0.6B"
# 本地目标目录
target_dir = r"./models/Qwen3-0.6B"

# 下载模型到本地（snapshot_download 会自动断点续传）
snapshot_download(
    repo_id=model_name,
    local_dir=target_dir,
    local_dir_use_symlinks=False  # 避免符号链接，保证真实文件落地
)
```

---

### 📌 加载模型与推理

模型下载完成后，可以直接从本地目录加载，避免重复联网：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从本地路径加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(target_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    target_dir,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# 测试推理
prompt = "请用简短的话介绍一下大语言模型的作用。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
 
---

# 模型生成部分总结

### 1. 加载 Tokenizer 和模型

```python
# 指定模型保存路径
local_dir = "./models/Qwen3-0.6B"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    torch_dtype="auto",
    device_map="cuda:0"
)
```

AutoModelForCausalLM → 加载一个因果语言模型（CausalLM）。

torch_dtype="auto" → 自动选择浮点精度，节省显存。

device_map="auto" → 自动分配显存到 GPU/CPU。

---

### 2. 构造输入文本

```python
messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 启用思考模式
)
```

此时输入会被转化为以下形式：

```
<|im_start|>system
SYS_PROMPT
<|im_end|>
<|im_start|>user
prompt
<|im_end|>
<|im_start|>assistant
```

这种标注方式能让大模型更好地理解**多轮对话的角色和轮次**。最后的 `<|im_start|>assistant` 表明：现在轮到模型回答。

但从另一个方面，我们可以get其实多轮对话，到了输入这一环节，其实也是整理合并为一条很长的字符串，这也是要进行messages管理防止幻觉强调主旨的原因。

---

### 3. Tokenizer 将文字转化为 token

```python
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
```

输出示例：

```
{
  'input_ids': tensor([[151644,   8948, ...]], device='cuda:0'),
  'attention_mask': tensor([[1, 1, ...]], device='cuda:0')
}
```

---

### 4. 模型生成

```python
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
```

结果是一个 **token 序列**，如：

```
tensor([[151644,   8948,   271, ...,   532, 73594, 151645]], device='cuda:0')
```

---

### 5. 截取输出部分

只取新增的输出（输入部分不需要重复）：

```python
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
```

---

### 6. 解析输出内容

模型可能包含 **thinking** 和 **content** 两部分，需要分开解码：

```python
try:
    # 找到 </think> 的 token 位置（反转搜索，保证取最后一次出现）
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
```

---
## ✅ 总结

1. **下载**：`snapshot_download` 可断点续传，推荐国内环境使用。
2. **镜像**：设置 `HF_ENDPOINT=https://hf-mirror.com`，大幅提高下载速度和成功率。
3. **缓存路径**：通过 `HF_HOME` 控制模型存储目录，避免模型散落在默认缓存路径。
4. **加载**：下载一次后，直接从本地目录加载即可，**完全离线推理**。

---
