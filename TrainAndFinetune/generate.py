from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== #
# 1. 模型与 Tokenizer 加载
# ===================== #
LOCAL_DIR = "./models/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR,
    torch_dtype="auto",
    device_map="cuda:0"
)

# ===================== #
# 2. 系统提示词
# ===================== #
SYS_PROMPT = """
你是一位**专业的医疗知识抽取助手**。
我将提供一段或多段**医疗相关文本**，你的任务是：

1. 从文本中抽取与药品相关的信息。
2. 按照给定的 JSON Schema 填充字段。
3. 如果某个字段在文本中未出现，请置为空字符串 `""`。
4. 严格输出为 JSON 格式，不要添加多余说明。

### JSON Schema 定义
{
  "药品名称": "药品的通用名或商品名",
  "价格参考": "药品的价格信息，例如单价、区间或剂型价格",
  "中国上市情况": "是否已在中国获批上市，或临床阶段信息",
  "靶点": "药物的作用靶点，例如 PD-1、VEGFR 等",
  "适应症": "药品主要的适应症或获批/研究适应症",
  "参考用法用量": "药品推荐的用法与剂量",
  "不良反应": "常见或已报道的不良反应"
}

### 输出要求
* 严格遵循 JSON 格式，不要输出额外文字。
* 所有字段都必须出现，若缺失填 `""`。
"""

# ===================== #
# 3. 用户输入
# ===================== #
prompt = """
雷莫芦单抗是一种**靶向血管生成**的单克隆抗体药物，由美国礼来公司研发，主要用于治疗多种晚期实体瘤。
近年来，其在中国也以新名称“雷莫西尤单抗”获批上市（商品名：希冉择®）。

---
#### 一、基本信息
| 项目 | 内容 |
|------|------|
| 药品通用名 | 雷莫芦单抗 / 雷莫西尤单抗（Ramucirumab） |
| 商品名 | Cyramza（国外）、希冉择®（中国） |
| 生产企业 | 礼来制药（Lilly） |
| 是否中国上市 | **已上市**（2023年批准，商品名为“希冉择®”） |
| 规格 | 100mg/支、500mg/支 |

> 注：此前因未在国内获批，患者多通过海外代购获取，价格较高。

---
#### 二、作用机制与靶点
- **靶点**：**VEGFR-2**
- **作用机制**：
  - 能特异性结合 VEGFR-2，阻断其与配体结合。
  - 抑制肿瘤血管新生，切断血液供应。

✅ 推荐剂量：150 mg 口服，每天 2 次，在进餐前 1 小时或餐后 2 小时服用。

副作用提示：
- 常见不良反应（达拉非尼单药）：头痛、发热、关节炎、脱发等。
- 达拉非尼+曲美替尼：畏寒、疲乏、皮疹、恶心、腹泻、关节痛等。
"""

# ===================== #
# 4. 构造输入
# ===================== #
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

print(f"[DEBUG] 编码前文本:\n{text}\n")

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(f"[DEBUG] 模型输入:\n{model_inputs}\n")

# ===================== #
# 5. 模型推理
# ===================== #
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

print(f"[DEBUG] 生成的 tokens:\n{generated_ids}\n")

# 只取新增的输出部分
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
print(f"[DEBUG] 输出 tokens:\n{output_ids}\n")

# ===================== #
# 6. 结果解析
# ===================== #
END_THINK_TOKEN_ID = 151668  # </think>

try:
    # 找到 </think> 的位置（反向搜索确保取最后一个）
    index = len(output_ids) - output_ids[::-1].index(END_THINK_TOKEN_ID)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

print("========== 模型思考 ==========")
print(thinking_content)
print("========== 模型输出 ==========")
print(content)
