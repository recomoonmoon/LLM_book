### 大模型基础
* 教程：
  * [大模型基础视频教程](https://www.bilibili.com/video/BV1Bo4y1A7FU/)

* 环境依赖
  * openai 调用API
  * dotenv 存储
  
* 主要内容
  * [Prompt Engineer](./1_prompt.md)
  * [LangChain](./2_lanchain.md)

---

# LangChain Prompt 用法总结

Prompt 是大模型（LLM）的输入，可以是简单字符串，也可以是多轮对话消息。在 LangChain 中，Prompt 的设计方式影响模型的输出效果。

---

## 一、PromptTemplate（单轮字符串模板）

* **用途**：构建固定格式的字符串 prompt，变量用 `{key}` 占位符。
* **创建方式**：

  1. `PromptTemplate.from_template(template)`
  2. `PromptTemplate(template=..., input_variables=[...])`
* **调用方式**：

  * `.format(**vars)`
  * 串联到 `chain` 使用

**示例：**

```python
prompt = "---{disease}---有---{symptom}---症状，需要使用---{medicine}---药品进行治疗"
var_dict = {"disease": "糖尿病", "symptom": "尿血", "medicine": "格列美脲"}

prompt_template = PromptTemplate.from_template(prompt)
print(prompt_template.format(**var_dict))
```

---

## 二、FewShotPromptTemplate（示例提示）

* **用途**：通过给定输入输出示例，引导模型学习格式。
* **关键点**：

  * `examples`：样例列表
  * `example_prompt`：样例格式
  * `suffix`：留出用户输入的位置

**示例：**

```python
examples = [
    {"word": "cat", "translation": "猫"},
    {"word": "dog", "translation": "狗"}
]
example_prompt = PromptTemplate.from_template("英文: {word} -> 中文: {translation}")

fewshot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="英文: {word} -> 中文:",
    input_variables=["word"]
)
print(fewshot_prompt.format(word="apple"))
```

---

## 三、ChatPromptTemplate（多轮对话）

* **用途**：模拟对话场景，由 system / human / ai 消息构成。
* **常见场景**：问答助手、任务型对话。

**示例：**

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个医学助手。"),
    ("human", "病人患有{disease}，出现了{symptom}，应该如何治疗？")
])
print(chat_prompt.format(disease="糖尿病", symptom="尿血"))
```

---

## 四、MessagePromptTemplate（精细化消息控制）

* **用途**：精确指定消息角色，例如 System/Human/AI。
* **适合场景**：需要控制角色语气或功能时。

**示例：**

```python
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个翻译助手。"),
    HumanMessagePromptTemplate.from_template("请翻译这句话: {sentence}")
])
print(chat_prompt.format(sentence="我今天很开心"))
```

---

## 五、MessagesPlaceholder（插入对话历史）

* **用途**：在 Prompt 中动态插入对话历史，实现记忆功能。
* **常用于**：多轮对话，带上下文记忆。

**示例：**

```python
chat_with_memory = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "请继续回答: {question}")
])
history = [
    {"role": "human", "content": "你好"},
    {"role": "ai", "content": "你好，我能帮你什么？"}
]
print(chat_with_memory.format(history=history, question="今天天气怎么样？"))
```

---

好的 👍 我来模仿之前的 **Prompt 部分 README 笔记风格**，帮你整理一份关于 **LangChain Parser（输出解析器）** 的说明，结合示例代码，保持条理清晰：

---

# LangChain Parser 用法总结

在 LangChain 中，**Parser（解析器）** 用于将大模型输出的 **非结构化文本** 转换为 **结构化结果**（如 JSON、字典、表格等），方便后续程序使用。

---

## 一、ResponseSchema（定义输出字段）

* **用途**：定义希望模型输出的字段名和说明。
* **写法**：为每个字段创建 `ResponseSchema` 对象。

**示例：**

```python
from langchain.output_parsers import ResponseSchema

# 定义输出结构
response_schemas = [
    ResponseSchema(name="disease", description="疾病名称"),
    ResponseSchema(name="symptom", description="相关症状"),
    ResponseSchema(name="medicine", description="推荐药物")
]
```

---

## 二、StructuredOutputParser（结构化解析器）

* **用途**：告诉模型必须严格按指定格式输出，并自动解析成 Python 字典。
* **结合 ResponseSchema 使用**：

  1. `StructuredOutputParser.from_response_schemas(...)`
  2. 获取 `format_instructions`，拼进 Prompt，引导模型输出 JSON 格式。

**示例：**

```python
from langchain.output_parsers import StructuredOutputParser

# 创建解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

print("格式说明：", format_instructions)
```

输出示例（模型被要求返回这种 JSON）：

```json
{
  "disease": "...",
  "symptom": "...",
  "medicine": "..."
}
```

---

## 三、Parser 与 Prompt 配合

将 `format_instructions` 加入 Prompt，引导模型严格遵守输出格式。

**示例：**

```python
from langchain_core.prompts import PromptTemplate

template = """
你是一个医学助手。
请根据信息生成 JSON 输出。
信息：疾病={disease}，症状={symptom}，推荐药物={medicine}

输出要求：
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["disease", "symptom", "medicine"],
    partial_variables={"format_instructions": format_instructions}
)

final_prompt = prompt.format(
    disease="糖尿病", symptom="尿血", medicine="格列美脲"
)
print(final_prompt)
```

---

## 四、解析模型返回结果

模型调用后，先得到原始字符串，再通过 `output_parser.parse(...)` 转换为结构化结果。

**示例：**

```python
# 模拟大模型输出
llm_output = '{"disease": "糖尿病", "symptom": "尿血", "medicine": "格列美脲"}'

parsed_result = output_parser.parse(llm_output)
print(parsed_result)   # {'disease': '糖尿病', 'symptom': '尿血', 'medicine': '格列美脲'}
```

---

## 五、Parser 技巧总结

1. **约束输出格式**：在 Prompt 中加入 `format_instructions`，强制模型生成标准 JSON。
2. **错误处理**：解析失败时可加 try/except 捕获，避免程序崩溃。
3. **结合链式调用**：Prompt → LLM → Parser → 结构化结果，一步到位。
4. **可扩展性**：换不同的 `ResponseSchema` 就能快速切换输出字段。

---

👉 这样，Parser 让 LLM 输出从 “随意的自然语言” 变成 “严格的结构化结果”，便于自动化处理。

---
 


