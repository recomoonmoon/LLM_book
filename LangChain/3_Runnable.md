 
---

# ⚙️ LangChain Runnable 教程笔记

## 📑 目录

* [1. 什么是 Runnable](#1-什么是-runnable)
* [2. `invoke`（单次调用）](#2-invoke单次调用)
* [3. `batch`（并行处理）](#3-batch并行处理)
* [4. `batch_as_completed`（乱序并行）](#4-batch_as_completed乱序并行)
* [5. 异步调用（async）](#5-异步调用async)
* [6. 流式调用（stream / astream）](#6-流式调用stream--astream)
* [7. 输入和输出类型](#7-输入和输出类型)
* [8. RunnableConfig（运行时配置）](#8-runnableconfig运行时配置)
* [9. 自定义 Runnable](#9-自定义-runnable)
* [10. 可配置 Runnables](#10-可配置-runnables)
* [11. Streaming APIs（流式接口）](#11-streaming-apis流式接口)
* [12. Inspecting Schemas（检查输入输出结构）](#12-inspecting-schemas检查输入输出结构)
* [13. with\_types（手动指定类型）](#13-with_types手动指定类型)
* [14. RunnableConfig 传递（Propagation）](#14-runnableconfig-传递propagation)
* [15. 配置选项详解](#15-配置选项详解)

  * [15.1 run\_name / run\_id](#151-run_name--run_id)
  * [15.2 tags / metadata](#152-tags--metadata)
  * [15.3 recursion\_limit](#153-recursion_limit)
  * [15.4 max\_concurrency](#154-max_concurrency)
  * [15.5 configurable](#155-configurable)
  * [15.6 callbacks](#156-callbacks)
* [16. 从函数创建 Runnable](#16-从函数创建-runnable)
* [17. 可配置 Runnables（Configurable Runnables）](#17-可配置-runnablesconfigurable-runnables)
* [18. 总结](#18-总结)

---

## 1. 什么是 Runnable

`Runnable` 是 **LangChain 的统一执行接口**，几乎所有核心组件（LLM、ChatModel、Prompt、Retriever、OutputParser、LangGraph 等）都实现了它。

### 常用方法

* `invoke`：单输入 → 单输出
* `batch`：多个输入并行处理 → 多个输出
* `stream`：流式输出
* `async`：异步调用
* `compose`：组合多个 Runnable

### 示例

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
result = llm.invoke("Hello, how are you?")
print(result.content)
```

---

## 2. `invoke`（单次调用）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

# 单次调用
resp = llm.invoke("请用一句话总结：人工智能的核心价值")
print(resp.content)
```

---

## 3. `batch`（并行处理）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

# 并行处理多个输入
questions = ["什么是AI？", "量子计算的前景？", "太阳能的优势？"]
responses = llm.batch(questions)

for r in responses:
    print(r.content)
```

---

## 4. `batch_as_completed`（乱序并行）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

questions = ["A", "B", "C"]

for resp in llm.batch_as_completed(questions):
    print(resp.content)  # 按完成顺序返回，而不是输入顺序
```

---

## 5. 异步调用（async）

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

async def main():
    result = await llm.ainvoke("什么是机器学习？")
    print(result.content)

asyncio.run(main())
```

---

## 6. 流式调用（stream / astream）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

# 同步流式输出
for chunk in llm.stream("请逐步解释深度学习的基本原理"):
    print(chunk.content, end="", flush=True)
```

---

## 7. 输入和输出类型

```python
print("输入模式:", llm.input_schema.schema())
print("输出模式:", llm.output_schema.schema())
```

---

## 8. RunnableConfig（运行时配置）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

config = {"tags": ["demo"], "metadata": {"topic": "AI"}}
resp = llm.invoke("解释大语言模型的用途", config=config)
print(resp.content)
```

---

## 9. 自定义 Runnable

```python
from langchain_core.runnables import RunnableLambda

# 自定义一个翻倍函数
double = RunnableLambda(lambda x: x * 2)
print(double.invoke(10))  # 20
```

---

## 10. 可配置 Runnables

```python
from langchain_core.runnables import RunnableConfigurableFields
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

configurable_llm = RunnableConfigurableFields(
    llm, model="gpt-4o-mini", temperature=0.7
)

resp = configurable_llm.invoke("简述强化学习")
print(resp.content)
```

---

## 11. Streaming APIs（流式接口）

* `stream / astream`：常见流式输出
* `astream_events`：高级事件流
* `astream_log`：旧版 API

---

## 12. Inspecting Schemas（检查输入输出结构）

```python
print("输入:", llm.get_input_schema())
print("输出:", llm.get_output_schema())
print("配置:", llm.config_schema())
```

---

## 13. with\_types（手动指定类型）

```python
chain = (llm).with_types(input_type=dict, output_type=str)
```

---

## 14. RunnableConfig 传递（Propagation）

* LCEL 组合：自动传播配置
* Python 3.9/3.10 异步：需手动传递

---

## 15. 配置选项详解

### 15.1 run\_name / run\_id

```python
config = {"run_name": "demo", "run_id": "1234"}
llm.invoke("介绍机器学习", config=config)
```

### 15.2 tags / metadata

```python
config = {"tags": ["education"], "metadata": {"course": "AI"}}
```

### 15.3 recursion\_limit

限制递归层数，防止无限循环。

### 15.4 max\_concurrency

设置最大并发数。

### 15.5 configurable

传递运行时参数。

### 15.6 callbacks

设置回调函数。

---

## 16. 从函数创建 Runnable

```python
from langchain_core.runnables import RunnableLambda

def square(x): return x * x
square_runnable = RunnableLambda(square)
print(square_runnable.invoke(5))  # 25
```

---

## 17. 可配置 Runnables（Configurable Runnables）

```python
from langchain_core.runnables import ConfigurableField

configurable_llm = llm.configurable_fields(
    model=ConfigurableField(id="llm_model")
)

resp = configurable_llm.invoke("解释深度强化学习", config={"configurable": {"llm_model": "gpt-4"}})
print(resp.content)
```

---

## 18. 总结

* **Runnable = 统一执行接口**
* 常用方法：`invoke / batch / stream / async`
* **配置灵活**：`RunnableConfig`
* **高级功能**：Streaming、Schema 检查、Configurable Runnables

👉 Runnable 是 **LangChain 的核心执行抽象**，掌握它就能自由组合和扩展各种链式任务。

---

 