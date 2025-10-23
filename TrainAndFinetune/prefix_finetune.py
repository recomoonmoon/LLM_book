import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import PrefixTuningConfig, get_peft_model

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

peft_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=30  # 训练的前缀长度
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

#数据准备
import json
from datasets import Dataset

# 加载数据
data_list = json.load(open("./dataset/urban_factory_dataset.json"))

# 构建 HuggingFace Dataset
dataset = Dataset.from_list(data_list)


def process_func(data):
    MAX_LENGTH = 4096
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
    response = tokenizer(data["output"] + tokenizer.eos_token, add_special_tokens=False) #加上eos token防止微调后重复输出

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    #截断，虽然一般要小心截断eos
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

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


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=peft_model,
    padding=True
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

peft_model.save_pretrained("./output_prefix_qwen")
tokenizer.save_pretrained("./output_prefix_qwen")

print("Prefix Tuning 完成！模型已保存到 ./output_prefix_qwen")
