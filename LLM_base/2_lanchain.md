
---

# 📘 LangChain + Tongyi 使用笔记

## 📖 参考资料

* [LangChain 中文文档](https://python.langchain.ac.cn/docs/introduction/)
* [吴恩达《LLM Cookbook》教程](https://datawhalechina.github.io/llm-cookbook/#/C2/readme)

---

## 🚀 环境准备

在国内环境下，不使用 OpenAI 的 key，而是使用 **通义千问 (Qwen / Tongyi)** 的 API key。

1. 在 `record.env` 文件中配置环境变量：
   ```env
   QWEN_API_KEY=your_qwen_key
   QWEN_URL=your_qwen_url
   ```

2. 在代码里加载并设置：

   ```python
   from dotenv import load_dotenv
   import os

   if load_dotenv("../record.env"):
       os.environ["DASHSCOPE_API_KEY"] = os.environ["QWEN_API_KEY"]
   ```

---

## ⚙️ 基础调用

新版 LangChain 已弃用 `langchain.llms`，应使用 `langchain_community.llms.Tongyi`：

```python
from langchain_community.llms import Tongyi

llm = Tongyi(model="qwen-turbo", temperature=0.7)
print(llm.invoke("hello world"))
```

---

## 🔗 使用 PromptTemplate + Chain

封装一个简单的函数，把 **模板** 和 **模型** 串成一个 chain：

```python
def invoke_and_chain(model, temperature, prompt_p, var_dict):
    client = Tongyi(model=model, temperature=temperature)
    prompt = PromptTemplate.from_template(prompt_p)
    chain = prompt | client
    return chain.invoke(var_dict)
```

调用示例：

```python
model = "qwen-turbo"
temperature = 0.7
prompt_p = '''
{var1} * {var2} 等于多少？
'''
vardict = {"var1": 5, "var2": 7}

response = invoke_and_chain(
    model=model, 
    temperature=temperature, 
    prompt_p=prompt_p, 
    var_dict=vardict
)
print(response)
```

输出：

```
35
```

---

## 💡 提示技巧

* `PromptTemplate` 支持变量插值，可快速复用模板。
* 使用 `"Let's think step by step"` 等思维链提示，可以提升推理准确率。
* 对于对话式场景，可改用 `ChatTongyi` + `SystemMessage/HumanMessage`。

---

 