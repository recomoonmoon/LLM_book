from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型保存路径
local_dir = "./models/Qwen3-0.6B"

#加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    torch_dtype="auto",
    device_map="cuda:0"
)

# 构造输入
SYS_PROMPT = """
你是一位**专业的医疗知识抽取助手**。
我将提供一段或多段**医疗相关文本**，你的任务是：

1. 从文本中抽取与药品相关的信息。
2. 按照给定的 JSON Schema 填充字段。
3. 如果某个字段在文本中未出现，请置为空字符串 `""`。
4. 严格输出为 JSON 格式，不要添加多余说明。

### JSON Schema 定义

```json
{
  "药品名称": "药品的通用名或商品名",
  "价格参考": "药品的价格信息，例如单价、区间或剂型价格",
  "中国上市情况": "是否已在中国获批上市，或临床阶段信息",
  "靶点": "药物的作用靶点，例如 PD-1、VEGFR 等",
  "适应症": "药品主要的适应症或获批/研究适应症",
  "参考用法用量": "药品推荐的用法与剂量",
  "不良反应": "常见或已报道的不良反应"
}
```

### 输出要求

* 严格遵循 JSON 格式，不要输出额外文字。
* 所有字段都必须出现，若缺失填 `""`。

 
"""
prompt = """
 雷莫芦单抗是一种**靶向血管生成**的单克隆抗体药物，由美国礼来公司（Eli Lilly）研发，主要用于治疗多种晚期实体瘤。近年来，其在中国也以新名称“**雷莫西尤单抗**”获批上市（商品名：希冉择®），标志着该药在国内临床应用的推进。
---

#### 一、基本信息

| 项目 | 内容 |
|------|------|
| 药品通用名 | 雷莫芦单抗 / 雷莫西尤单抗（Ramucirumab） |
| 商品名 | Cyramza（国外）、希冉择®（中国） |
| 生产企业 | 礼来制药（Lilly） |
| 是否中国上市 | **已上市**（2023年批准，商品名为“希冉择®”） |
| 规格 | 100mg/支、500mg/支 |

> 注：此前因未在国内获批，患者多通过海外代购或特殊渠道获取土耳其版等，价格较高且不稳定。

---

#### 二、作用机制与靶点

- **靶点**：**VEGFR-2**（血管内皮生长因子受体-2）
- **作用机制**：
  - 雷莫芦单抗是一种全人源IgG1单克隆抗体，能特异性结合VEGFR-2，阻断其与配体（如VEGF-A、VEGF-C、VEGF-D）的结合。
  - 抑制肿瘤血管新生，从而切断肿瘤的血液供应，抑制肿瘤生长和转移。

✅ 属于**抗血管生成类靶向药**，不同于直接杀伤癌细胞的化疗或针对癌基因的靶向药。推荐剂量是150 mg口服每天2次在进餐前至少1小时或后至少2小时服用。

有一些点需要注意，尤其是副作用。达拉非尼单药，最常见达拉非尼不良反应(≥20%)是头痛，发热，关节炎，乳头状瘤，脱发，和掌跖红肿疼痛综合征。达拉非尼与曲美替尼联用，最常见不良反应(≥20%)包括发热，畏寒，疲乏，皮疹，恶心，呕吐，腹泻，腹痛，外周性水肿，咳嗽，头痛，关节痛，夜汗，食欲减低，便秘，和肌痛。"
  
"""
messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

print(f"text {text}")

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print(f"model_inputs {model_inputs}")
"""
 
"""


# 推理
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

print(f"generated_ids {generated_ids}")
""" 一个tensor的二维张量 """

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
'''
截取其中一维部分，从输入后面开始截取（输出= 输入 + 增加的输出）
'''
print(f"output_ids {output_ids}")

# 分析输出内容
try:
    #找到thinking部分的token位置 但为什么要反转？
    index = len(output_ids) - output_ids[::-1].index(151668)  # </think> 的 token id
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
