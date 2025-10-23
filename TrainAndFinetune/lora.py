# ========== 数据准备 ==========
import json
from datasets import Dataset





# 加载数据
data_list = [json.loads(line) for line in open("train.json", "r", encoding="utf-8")]

# 构建 HuggingFace Dataset
dataset = Dataset.from_list(data_list)





# 构建文本格式：简单拼接 prompt + output
def format_example(example):
    return f"指令：{example['instruction']}\n回答：{example['output']}"

def preprocess(example):
    text = format_example(example)
    tokenized = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ========== 训练参数 ==========
training_args = TrainingArguments(
    output_dir="./output_prefix_qwen",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-4,
    bf16=True,
    optim="adamw_torch",
    report_to="none",  # 不使用wandb
)

# ========== Data Collator ==========
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=peft_model,
    padding=True
)

# ========== Trainer ==========
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ========== 开始训练 ==========
trainer.train()

# ========== 保存模型（仅保存Prefix参数） ==========
peft_model.save_pretrained("./output_prefix_qwen")
tokenizer.save_pretrained("./output_prefix_qwen")

print("✅ Prefix Tuning 完成！模型已保存到 ./output_prefix_qwen")
