 
---

# 🔹 LoRA 微调 (Low-Rank Adaptation)

## 1. 什么是 LoRA？

LoRA（Low-Rank Adaptation of Large Language Models）是一种 **参数高效微调 (PEFT)** 方法。
它的核心思想是：

* 冻结大模型的原始参数（保持预训练知识不变）。
* 在注意力层（Q/K/V/Output 投影矩阵等）中插入 **低秩矩阵 A 和 B**。
* 训练时仅更新这些小规模的 **低秩参数**，从而大幅减少显存和存储开销。

其核心思想就是，权重都是一个个大矩阵，假设矩阵size是m * n的。
那就可以使用 m * k 和 k * n的矩阵相乘来拟合。

k就是我们配置参数里面的rank秩，r越大，拟合精度和效果就越好，占据显存也越大。

 

---

## 2. LoRA vs 全参数微调

| 对比项  | 全参数微调 (Full Finetune) | LoRA 微调            |
| ---- | --------------------- | ------------------ |
| 参数量  | 更新所有参数 (数十亿)          | 仅更新少量 LoRA 层参数     |
| 显存需求 | 极高 (需 A100 80GB)      | 极低 (10GB 甚至更少)     |
| 训练速度 | 较慢                    | 较快                 |
| 知识保留 | 可能遗忘预训练知识             | 保持原模型知识，专注新任务      |
| 模型大小 | finetune 后模型很大        | 原模型 + LoRA 权重 (很小) |

---

## 3. HuggingFace + PEFT 核心实现示例
其实lora部分和前面finetune差不多，但是要先配置一个lora参数设置。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,      # 指定模型类型（例如：因果语言模型 CLM）
    target_modules=["q_proj", "v_proj"], # 指定哪些模块要进行 LoRA 微调
    inference_mode=False,              # 设置是否仅用于推理
    r=4,                               # LoRA 的秩 (rank)
    lora_alpha=16,                     # LoRA 的缩放参数
    lora_dropout=0.05,                 # LoRA 的 Dropout 防止过拟合
    bias="none"                        # 偏置参数设置
)
model = get_peft_model(model, config)
model.print_trainable_parameters()     # 输出可训练参数数目

```

---

## 4. 使用方法

* 训练完成后，只保存 LoRA 的权重（通常只有几 MB）。
* 推理时加载原始大模型，再加载 LoRA 权重进行合并。
 

---

## 5. 适用场景

* 显存不足无法做全量微调。
* 数据集较小，只需在预训练模型上做轻量定制。
* 多任务 / 多领域训练（每个任务保存一个 LoRA 权重即可，方便切换）。

---

我可以帮你把这个 md 文件写成 **教程型（更详细步骤）** 还是 **概念型（更学术解释）** 的，你想要哪种风格？
