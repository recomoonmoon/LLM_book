好的✅ 我帮你把文档重新整理了一下，排版清晰、目录带链接、文字更简洁流畅，便于学习和查阅。

---

# 第五部分：训练与微调

我们正式进入 **第五部分：训练与微调**。

目前已有许多训练和微调工具，其中最著名的就是 **[LLaMA-Factory](https://llamafactory.readthedocs.io/zh-cn/latest/)**。它配置好文件和数据即可完成训练，非常好用。但为了学习目的，我们先从原理出发，理解如何进行训练和微调，再学习如何使用这些工具。

由于项目和就业场景主要依赖开源模型，本节重点以 **Qwen 系列模型** 为主进行讲解与实践。

---

## 📚 参考资料

* [LLaMA-Factory 官方文档](https://llamafactory.readthedocs.io/zh-cn/latest/)
* [Qwen3 快速入门文档](https://qwen.readthedocs.io/zh-cn/latest/getting_started/quickstart.html)

---

## 📑 目录

1. [环境准备](#环境准备)
2. [数据处理](#数据处理)
   * [Alpaca 格式](#alpaca-格式)
   * [ShareGPT 格式](#sharegpt-格式)
3. [模型的下载与推理](#模型的下载与推理)
4. [预训练](#预训练)
5. [微调](#微调)

---

## 环境准备

* Python ≥ 3.10
* PyTorch ≥ 2.6
* `transformers` ≥ 4.51.0

---

## 数据处理

数据的核心在于 **`dataset_info.json`**，它定义了所有数据集（本地 & 在线）的信息。

👉 如果你需要使用自定义数据集，请务必在 `./dataset/dataset_info.json` 中添加相应描述。

目前支持两大主要格式：

* **Alpaca 格式**
* **ShareGPT 格式**

此外，还支持 **多模态数据集**（图像/视频/音频）、**偏好数据集**、**KTO 数据集** 等。

---

### Alpaca 格式

Alpaca 格式适合 **指令监督微调**、**预训练**、**偏好训练**、**KTO**、**多模态任务**。

#### 1. 指令监督微调

模型学习输入指令与输出回答的对应关系。
基本字段：

* `instruction`（必填）：人类指令
* `input`（选填）：额外输入
* `output`（必填）：模型回答
* `system`（选填）：系统提示词
* `history`（选填）：多轮对话历史

**示例：**

```json
[
  {
    "instruction": "计算这些物品的总费用。",
    "input": "汽车 - $3000，衣服 - $100，书 - $20。",
    "output": "总费用为 $3120。",
    "history": [
      ["今天会下雨吗？", "不会，下雨机率为0。"],
      ["适合出门吗？", "很适合。"]
    ]
  }
]
```

**对应 `dataset_info.json`：**

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

#### 2. 预训练数据集

```json
[
  {"text": "这是第一段训练文本。"},
  {"text": "这是第二段训练文本。"}
]
```

`dataset_info.json`：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}
```

#### 3. 偏好数据集

用于 **奖励建模**、**DPO/ORPO 训练**。

```json
[
  {
    "instruction": "人类指令",
    "chosen": "更优的回答",
    "rejected": "较差的回答"
  }
]
```

#### 4. KTO 数据集

```json
[
  {
    "instruction": "人类指令",
    "output": "模型回答",
    "kto_tag": true
  }
]
```

#### 5. 多模态数据集

支持 **图像 / 视频 / 音频** 输入，需在 JSON 中加入 `images` / `videos` / `audios` 字段。

---

### ShareGPT 格式

ShareGPT 格式相比 Alpaca 更灵活，支持更多角色（`human`、`gpt`、`function_call`、`observation` 等）。

#### 1. 指令监督微调

```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "gpt", "value": "你好，很高兴见到你！"}
  ]
}
```

#### 2. 偏好数据集

```json
{
  "conversations": [
    {"from": "human", "value": "老虎吃草吗？"}
  ],
  "chosen": {"from": "gpt", "value": "老虎是食肉动物。"},
  "rejected": {"from": "gpt", "value": "老虎主要吃草。"}
}
```

#### 3. OpenAI 格式

一种特殊情况，`messages` 字段中包含 `system`、`user`、`assistant`：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "今天天气怎么样？"},
    {"role": "assistant", "content": "今天是晴天。"}
  ]
}
```

 
---

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

### ✅ 总结

1. **下载**：`snapshot_download` 可断点续传，推荐用于国内环境。
2. **镜像**：设置 `HF_ENDPOINT=https://hf-mirror.com`，可大幅提高下载成功率。
3. **缓存路径**：通过 `HF_HOME` 控制模型存储位置，避免模型散落在默认缓存目录。
4. **加载**：下载一次后，直接从本地目录加载模型进行推理，完全离线运行。

---

 
---

## 预训练

介绍如何准备大规模未标注数据进行预训练。

---

## 微调

介绍如何基于指令数据集进行监督微调（SFT）、偏好建模（DPO/ORPO）、KTO 等。

---

 