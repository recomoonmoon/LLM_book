
---

# 🔹 LoRA 微调 (Low-Rank Adaptation)

## 参考资料
* [LoRA 微调与推理 从零开始的超详细教程](https://blog.csdn.net/2301_79996254/article/details/146296443)
* [对齐全量微调！这是我看过最精彩的LoRA改进](https://kexue.fm/archives/10226)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
---

## 1. 什么是 LoRA？

LoRA（Low-Rank Adaptation of Large Language Models）是一种 **参数高效微调 (PEFT)** 方法。
它的核心思想是：

* 冻结大模型的原始参数（保持预训练知识不变）。
* 在注意力层（Q/K/V/Output 投影矩阵等）中插入 **低秩矩阵 A 和 B**。
* 训练时仅更新这些小规模的 **低秩参数**，从而大幅减少显存和存储开销。

换句话说：
假设原始权重矩阵大小是 m * n，LoRA 用 m * r 和 r * n 两个小矩阵相乘来近似它。
其中 **r = rank 秩**，r 越大，拟合精度越高，但显存开销也会增加。

现在有很多LoRa的研究，lora初始weight和全量化一样，那么理想思路就是拟合全量化微调的梯度方向，推荐大家看看LoRa-GA这篇文献。

---

## 2. LoRA vs 全参数微调

| 对比项      | 全参数微调 (Full Finetune) | LoRA 微调            |
| -------- | --------------------- | ------------------ |
| **参数量**  | 更新所有参数 (数十亿)          | 仅更新少量 LoRA 层参数     |
| **显存需求** | 极高 (需 A100 80GB)      | 极低 (10GB 甚至更少)     |
| **训练速度** | 较慢                    | 较快                 |
| **知识保留** | 可能遗忘预训练知识             | 保持原模型知识，专注新任务      |
| **模型大小** | finetune 后模型很大        | 原模型 + LoRA 权重 (很小) |

---

## 2.1 📌 LoRA 参数详解

| 参数名                | 解释                                                                                                                     | 推荐值                      |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| **task_type**      | 指定任务类型<br>🔹 `TaskType.CAUSAL_LM`（GPT、Qwen）<br>🔹 `TaskType.SEQ2SEQ_LM`（T5）                                            | ✅ `TaskType.CAUSAL_LM`   |
| **target_modules** | 指定需要微调的 Transformer 模块<br>🔹 `q_proj`, `k_proj`, `v_proj` 控制注意力机制<br>🔹 `o_proj`, `gate_proj` 控制输出<br>💡 选择较少模块可减少显存占用 | ✅ `["q_proj", "v_proj"]` |
| **inference_mode** | 🔹 `False`：训练模式<br>🔹 `True`：推理模式（冻结）                                                                                  | ✅ `False`                |
| **r（秩）**           | 低秩分解的维度<br>r 越大，效果越好，但显存开销更大                                                                                           | ✅ 4 ~ 32                 |
| **lora_alpha**     | 缩放因子<br>`lora_alpha / r` 决定最终权重大小                                                                                      | ✅ 8 ~ 32                 |
| **lora_dropout**   | Dropout 概率，防止过拟合                                                                                                       | ✅ 0.05 ~ 0.1             |
| **bias**           | 偏置参数更新策略<br>🔹 `none`（不更新，推荐）<br>🔹 `all`（全部更新）<br>🔹 `lora_only`（仅 LoRA 层）                                            | ✅ `"none"`               |

---

## 2.2 🔎 参数选择经验

| 设备配置    | 模型大小 | 推荐参数                                   |
| ------- | ---- | -------------------------------------- |
| 8GB 显卡  | 7B   | r=4, lora_alpha=16, lora_dropout=0.05  |
| 12GB 显卡 | 13B  | r=8, lora_alpha=32, lora_dropout=0.05  |
| 24GB 显卡 | 30B  | r=16, lora_alpha=32, lora_dropout=0.05 |

---

## 3. HuggingFace + PEFT 核心实现示例

其实 LoRA 和全参数微调流程类似，只是多了一个 **LoRA 配置**。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载基础模型
model_name = "facebook/opt-1.3b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 配置 LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,       # 因果语言模型
    target_modules=["q_proj", "v_proj"],# 指定注入 LoRA 的模块
    inference_mode=False,               # 训练模式
    r=4,                                # 秩
    lora_alpha=16,                      # 缩放参数
    lora_dropout=0.05,                  # Dropout
    bias="none"                         # 不更新偏置
)

# 3. 应用 LoRA
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

---

## 4. 使用方法

* **训练后**：只保存 LoRA 权重（几 MB）。
* **推理时**：加载基础大模型，再加载 LoRA 权重进行合并。

---

## 5. 适用场景

* 显存不足，无法进行全量微调。
* 数据集较小，只需在预训练模型上做轻量定制。
* 多任务 / 多领域训练：每个任务保存一个 LoRA 权重，方便切换。

---

 