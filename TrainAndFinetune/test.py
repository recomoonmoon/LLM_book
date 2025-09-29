from transformers import AutoModelForCausalLM,AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen3_final",
    torch_dtype="auto",
    device_map="cuda:0"
)1
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-0.6B")

text = "接下来我会输入都市类小说的一小段剧情细纲，你需要理解剧情细纲的情节，将其扩充并润色到原本的2-3倍字数，形成可以直接给读者阅读的网络小说片段，然后将结果返回。"
chunk = "清见琉璃焦急地在中庭寻找最后一名新生，只为完成社团招新。她渴望成为“演绎推理精研社”首任部长，让全校倾倒。当她发现一个高瘦文质彬彬的一年级男生时，认定他不会加入运动社团，正是理想人选。"

messages = [
    {"role": "system", "content":text},
    {"role": "user", "content":chunk}
]
input_col = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
print(input_col)
input_col = tokenizer([input_col], return_tensors="pt").to(model.device)
print(input_col)
generated_ids = model.generate(
    **input_col,
    max_new_tokens=1000,
)
print("generate")
print(f"[DEBUG] 生成的 tokens:\n{generated_ids}\n")

# 只取新增的输出部分
output_ids = generated_ids[0].tolist()
print(f"[DEBUG] 输出 tokens:\n{output_ids}\n")

# ===================== #
# 6. 结果解析
# ===================== #


content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()


print("========== 模型输出 ==========")
print(content)