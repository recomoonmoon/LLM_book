import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

#lora 和finetune差不多，使用peft，和finetune比起来，主要是配置lora参数，get_peft_model替换模型

torch.cuda.empty_cache()

# ========== 模型与分词器 ==========
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="./models/Qwen3-0.6B",
    use_fast=False,
    trust_remote_code=True,
    padding_side="right"
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="./models/Qwen3-0.6B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ========== 注入 LoRA ==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 因果语言建模
    r=8,                           # LoRA rank
    lora_alpha=32,                 # LoRA scaling
    lora_dropout=0.05,             # dropout
    target_modules=["q_proj", "v_proj"]  # 常见选择: 只在注意力投影层插 LoRA
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数数量

# ========== 数据准备 ==========
dataset = Dataset.from_json("./dataset/urban_factory_dataset.json")

def process_func(data):
    MAX_LENGTH = 512
    system_prompt = (
        "<|im_start|>system\n"
        "你是我的都市小说助手。我会给你剧情片段或大纲，"
        "请你根据输入扩写成都市风格的小说片段，"
        "要求语言生动、贴近生活。\n"
        "<|im_end|>\n"
    )
    user_prompt = f"<|im_start|>user\n{data['instruction']}{data['input']}\n<|im_end|>\n"
    assistant_prompt = "<|im_start|>assistant\n"

    instruction = tokenizer(system_prompt + user_prompt + assistant_prompt, add_special_tokens=False)
    response = tokenizer(data["output"] + tokenizer.eos_token, add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)

# ========== 训练配置 ==========
args = TrainingArguments(
    output_dir="./models/Qwen3_lora_ft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-4,       # LoRA 一般可以用更大学习率
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    optim="adamw_torch_fused",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

trainer.train()

# 保存 LoRA adapter（不是全量模型）
model.save_pretrained("./models/Qwen3_lora_ft_final")
tokenizer.save_pretrained("./models/Qwen3_lora_ft_final")
