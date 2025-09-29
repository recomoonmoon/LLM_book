import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

torch.cuda.empty_cache()  # 清空显存，释放 GPU 资源

# ========== 模型与分词器 ==========
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="./models/Qwen3-0.6B",
    use_fast=False,  # 防止中文数据切分异常
    trust_remote_code=True,
    padding_side="right"  # 保证中文对齐时表现更稳定
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="./models/Qwen3-0.6B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # bfloat16 更省显存
    device_map="cuda"            # 自动分配到 GPU
)

# 关闭缓存并开启梯度检查点，减少显存消耗
model.config.use_cache = False
model.gradient_checkpointing_enable()


# ========== 数据准备 ==========
"""
数据格式:
[
  {"output": "...", "input": "...", "instruction": "..."}
]
"""
dataset = Dataset.from_json("./dataset/urban_factory_dataset.json")

def process_func(data):
    MAX_LENGTH = 512  # 你可以按显存情况调节

    # ====== prompt 模板 ======
    system_prompt = (
        "<|im_start|>system\n"
        "你是我的都市小说助手。我会给你剧情片段或大纲，"
        "请你根据输入扩写成都市风格的小说片段，"
        "要求语言生动、贴近生活。\n"
        "<|im_end|>\n"
    )

    user_prompt = (
        f"<|im_start|>user\n"
        f"{data['instruction']}{data['input']}\n"
        f"<|im_end|>\n"
    )

    assistant_prompt = "<|im_start|>assistant\n"

    # ====== tokenize ======
    instruction = tokenizer(
        system_prompt + user_prompt + assistant_prompt,
        add_special_tokens=False
    )
    response = tokenizer(
        data["output"] + tokenizer.eos_token,  # 结束符
        add_special_tokens=False
    )

    # 拼接
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]

    # 构造 labels：prompt 部分不计算 loss
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 数据映射
tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)


# ========== 训练配置 ==========
args = TrainingArguments(
    output_dir="./models/Qwen3_full_ft",  # 保存路径
    per_device_train_batch_size=2,       # batch size (根据显存调整)
    gradient_accumulation_steps=16,      # 梯度累积
    gradient_checkpointing=True,
    bf16=True,                           # bfloat16 (需硬件支持)
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,                      # 保存间隔大一点，减少I/O
    optim="adamw_torch_fused",
    max_grad_norm=0.3,
    warmup_ratio=0.03,                   # 预热学习率
    lr_scheduler_type="cosine",          # 余弦退火学习率
    report_to="tensorboard",             # 日志可视化
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),  # 动态padding，避免OOR
)

# 训练
trainer.train()

# 保存模型
trainer.save_model("./models/Qwen3_full_ft_final")
tokenizer.save_pretrained("./models/Qwen3_full_ft_final")
