
## 1. 什么是 Runnable

**概念**
`Runnable` 是 LangChain 的统一接口，几乎所有组件都实现了它（LLM、ChatModel、Prompt、Retriever、OutputParser、LangGraph Graph 等）。
它定义了一套标准方法，让我们用一致的方式去调用这些组件。

**核心能力**

* `invoke`：单输入 → 单输出
* `batch`：多个输入并行处理 → 多个输出
* `stream`：流式输出
* `inspect`：查看输入/输出类型和配置
* `compose`：把多个 Runnable 组合成 pipeline

**示例**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Runnable 的通用方法 invoke
result = llm.invoke("Hello, how are you?")
print(result.content)
```

---

## 2. `invoke`（单次调用）

* 最常用的方法：输入一个数据，返回一个结果。
* 输入和输出类型取决于具体组件。

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Translate this into French: {text}")
llm = ChatOpenAI(model="gpt-3.5-turbo")

chain = prompt | llm   # LCEL 组合成一个 runnable

print(chain.invoke({"text": "I love programming"}).content)
```

---

## 3. `batch`（并行处理）

* 处理多个输入，结果顺序与输入一致。
* 默认用线程池 → 提高 I/O 性能（比如调用 API）。

```python
texts = ["Hello", "How are you?", "Good night"]

results = llm.batch(texts)
for r in results:
    print(r.content)
```

---

## 4. `batch_as_completed`（乱序并行）

* 返回结果时不保证顺序，但会附带 index。
* 适合“先完成先返回”的场景。

```python
for idx, res in llm.batch_as_completed(texts):
    print(f"Input {idx} → {res.content}")
```

---

## 5. 异步调用（async）

* 所有主要方法都有异步版本（前缀 `a`）。

  * `ainvoke`
  * `abatch`
  * `astream`
  * `abatch_as_completed`

```python
import asyncio

async def main():
    result = await llm.ainvoke("Tell me a joke about AI")
    print(result.content)

asyncio.run(main())
```

---

## 6. 流式调用（stream / astream）

* 一边生成一边返回，提升用户体验。
* 常见于 LLM 输出。

```python
for chunk in llm.stream("Write a poem about the ocean."):
    print(chunk.content, end="")
```

---

## 7. 输入和输出类型

不同组件的输入输出类型不一样：

* **Prompt**：输入 dict → 输出 PromptValue
* **ChatModel**：输入 string/list → 输出 ChatMessage
* **LLM**：输入 string/list → 输出 string
* **Retriever**：输入 string → 输出 List\[Document]
* **OutputParser**：输入 LLM 输出 → 输出自定义类型

---

## 8. RunnableConfig（运行时配置）

调用 `.invoke()` 等方法时，可以传 `config`，控制运行参数。

```python
result = llm.invoke(
    "Hello",
    config={
        "run_name": "test_run",
        "tags": ["demo"],
        "metadata": {"topic": "greeting"}
    }
)
```

常用配置：

* `run_name`：运行名字
* `tags`：标签
* `metadata`：元数据
* `max_concurrency`：限制并行数量
* `callbacks`：回调函数（如日志、追踪）

---

## 9. 自定义 Runnable

如果你要插入自定义逻辑，可以用：

* `RunnableLambda`：简单函数包装
* `RunnableGenerator`：支持流式的函数包装

```python
from langchain_core.runnables import RunnableLambda

def to_upper(x: str) -> str:
    return x.upper()

upper_runnable = RunnableLambda(to_upper)

print(upper_runnable.invoke("hello"))  # 输出 "HELLO"
```

---

## 10. 可配置 Runnables

允许你在运行时调整配置，比如切换模型或改温度。

```python
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 通过 configurable_fields 暴露可调参数
llm = llm.configurable_fields("temperature")
```

---

📌 总结：
Runnable 就是 **LangChain 的统一执行接口**，核心方法是 `invoke / batch / stream / async`，再加上配置 `RunnableConfig` 和可组合性。

---
 
👌 好的，我们继续拆解 **Runnable 的知识点**。这次涉及到 **Streaming APIs → 输入输出类型 → Schema 检查 → RunnableConfig → 配置传播**。我会逐段讲解并给代码示例。

---

## 1. Streaming APIs（流式输出接口）

流式非常重要，它能让 LLM 应用对用户“即时响应”。Runnable 提供了 3 种流式接口：

1. **stream / astream**

   * 同步 `stream`，异步 `astream`。
   * 一边生成一边返回。

   ```python
   for chunk in llm.stream("Write a poem about the stars."):
       print(chunk.content, end="")
   ```

2. **astream\_events**

   * 高级 API，可以流式返回 **中间步骤 + 最终结果**。
   * 常用于复杂 chain/agent。

3. **astream\_log（已过时）**

   * 老版本的事件流 API，逐渐被 `astream_events` 替代。

👉 建议：一般用 `stream`/`astream`，需要中间步骤就用 `astream_events`。

---

## 2. 输入和输出类型（Input / Output Types）

每个 `Runnable` 都有输入类型和输出类型。

| 组件           | 输入类型                                 | 输出类型              |
| ------------ | ------------------------------------ | ----------------- |
| Prompt       | dict                                 | PromptValue       |
| ChatModel    | string / chat messages / PromptValue | ChatMessage       |
| LLM          | string / chat messages / PromptValue | string            |
| OutputParser | LLM 或 ChatModel 的输出                  | 自定义（如 dict, JSON） |
| Retriever    | string                               | List\[Document]   |
| Tool         | string 或 dict                        | 自定义               |

**示例**

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Translate: {text} → French")

# 输入是 dict
val = prompt.invoke({"text": "good morning"})
print(type(val))   # PromptValue
```

---

## 3. Inspecting Schemas（检查输入输出结构）

⚠️ 进阶功能，大多数人用不到，但在 **自动生成 API 文档 / 校验输入输出** 时有用。

Runnable 提供了 **Pydantic Schema 和 JSON Schema** 的查询方法：

* `get_input_schema()`
* `get_output_schema()`
* `config_schema()`
* `get_input_jsonschema()`
* `get_output_jsonschema()`
* `get_config_jsonschema()`

**示例**

```python
schema = llm.get_input_schema()
print(schema.schema_json(indent=2))
```

用途：

* 自动化测试
* LangServe 自动生成 OpenAPI 文档

---

## 4. with\_types（手动指定类型）

LangChain 会尝试推断 Runnable 的输入输出类型，但在 **复杂 LCEL 组合** 时可能出错。
这种情况下可以手动指定：

```python
chain = (prompt | llm).with_types(input_type=dict, output_type=str)
```

---

## 5. RunnableConfig（运行时配置）

几乎所有执行方法（`invoke/batch/stream/...`）都可以传一个 `config` dict。

常用参数：

* `run_name`: 运行名（仅父级，不继承）
* `run_id`: 自定义唯一 UUID（追踪）
* `tags`: 标签（会继承到子调用）
* `metadata`: 元数据（会继承到子调用）
* `callbacks`: 回调函数（日志/追踪）
* `max_concurrency`: 并行数限制
* `recursion_limit`: 避免无限递归
* `configurable`: 运行时传入的自定义参数

**示例**

```python
result = llm.invoke(
    "Hello",
    config={
        "run_name": "greeting_test",
        "tags": ["demo", "tutorial"],
        "metadata": {"topic": "intro"}
    }
)
```

---

## 6. Propagation of RunnableConfig（配置传播）

Runnable 可以由多个子 Runnable 组成（如 prompt | llm | parser）。
为了让子组件也能拿到 config（比如共享回调、metadata），LangChain 会自动传播 config。

**两种创建模式：**

1. **LCEL 声明式**

   ```python
   chain = prompt | llm | parser
   ```

   （自动传播 config）

2. **自定义 Runnable**（RunnableLambda / @tool）

   ```python
   from langchain_core.runnables import RunnableLambda

   def foo(input):
       return llm.invoke(input)

   foo_runnable = RunnableLambda(foo)
   ```

⚠️ Python 版本差异：

* **Python 3.11+**：自动支持 contextvars，无需手动传播。
* **Python 3.9 / 3.10（异步场景）**：需要手动传 `config`。

**示例（手动传播 config）**

```python
async def foo(input, config):
    return await llm.ainvoke(input, config=config)

foo_runnable = RunnableLambda(foo)
```

---

✅ 总结

* **Streaming APIs**：`stream / astream`（常用），`astream_events`（高级），`astream_log`（旧版）。
* **输入输出类型**：不同组件不同，Prompt → dict，LLM → string，ChatModel → ChatMessage。
* **Schema 检查**：`get_input_schema` 等方法，多用于测试/文档。
* **RunnableConfig**：运行时配置（run\_name, tags, metadata, callbacks…）。
* **配置传播**：在 LCEL 中自动传播，在 Python 3.9/3.10 异步需手动传。

---

 好的，我继续帮你整理成中文笔记，保持和前面一致的风格：

---

## ✅ RunnableConfig 的传递 (Propagation of RunnableConfig)

* 很多 **Runnables** 是由其他 **Runnables** 组成的，因此必须确保 **RunnableConfig** 可以传递到所有子调用 (sub-calls)。
* 这样做的好处是：可以在父 Runnable 中设置运行时配置（如 `callbacks`、`tags`、`metadata`），并自动继承到所有子调用。
* 如果不传递，将无法正确传播这些配置，导致调试和跟踪困难。

### 创建 Runnables 的两种方式

1. **声明式 (LCEL)**

   ```python
   chain = prompt | chat_model | output_parser
   ```

2. **自定义 Runnable**

   * 使用 `RunnableLambda` 或 `@tool` 装饰器

   ```python
   def foo(input):
       return bar_runnable.invoke(input)
   foo_runnable = RunnableLambda(foo)
   ```

LangChain 会自动尝试在两种方式下传播 `RunnableConfig`。
对于第二种方式，它依赖 **Python 的 contextvars**：

* **Python 3.11+**：默认支持，无需额外处理。
* **Python 3.9 / 3.10**：在异步 (async) 代码中，需要 **手动传递** `RunnableConfig`。

例如：

```python
async def foo(input, config):  # 注意这里接收 config
    return await bar_runnable.ainvoke(input, config=config)

foo_runnable = RunnableLambda(foo)
```

⚠️ **注意**：
在 **Python 3.10 或更低版本** 的异步环境中，`RunnableConfig` 无法自动传递，尤其在使用 `astream_events` 和 `astream_log` 时要小心。

---

## ✅ 设置自定义 Run 名称、标签、元数据

* 在 `RunnableConfig` 中，可以配置以下字段：

  * `run_name`：字符串，当前调用的自定义名称（不会继承给子调用）。
  * `tags`：列表，用于添加标签，会继承给子调用。
  * `metadata`：字典，用于添加元数据，会继承给子调用。

用途：

* 在 **LangSmith** 中可以用于调试、跟踪、过滤。
* 会在 **回调 (callbacks)** 和 **流式 API (astream\_events)** 中显示。

---

## ✅ 设置 run\_id

* 高级功能（大部分用户无需使用）。
* `run_id` 必须是 **UUID 字符串**，且对每个运行唯一。
* 用于唯一标识调用，便于跨系统关联。

示例：

```python
import uuid

run_id = uuid.uuid4()
some_runnable.invoke(
   some_input,
   config={"run_id": run_id}
)
```

---

## ✅ 设置递归限制 (recursion\_limit)

* 一些 Runnables 可能会返回新的 Runnables，导致递归调用。
* 为避免无限递归，可以在 `RunnableConfig` 中设置 `recursion_limit`。

---

## ✅ 设置最大并发 (max\_concurrency)

* 在 `batch` 或 `batch_as_completed` 中使用。
* 控制最大并行调用数，防止过载。

👉 更推荐使用 **LangChain 内置的速率限制器** 来处理请求速率，而不是 `max_concurrency`。

---

## ✅ 设置 configurable

* 用于传递运行时参数。
* 常见场景：

  * **LangGraph** 中的持久化与记忆功能。
  * **RunnableWithMessageHistory** 中指定 `session_id` 或 `conversation_id`。
  * 自定义参数传递给 Configurable Runnable。

---

## ✅ 设置 callbacks

* 可以在运行时配置回调，回调会传递给所有子调用。

示例：

```python
some_runnable.invoke(
   some_input,
   {
      "callbacks": [
         SomeCallbackHandler(),
         AnotherCallbackHandler(),
      ]
   }
)
```

---

## ✅ 从函数创建 Runnable

两种方式：

* `RunnableLambda`：适用于简单逻辑（不需要流式）。
* `RunnableGenerator`：适用于需要流式的复杂逻辑。

👉 不推荐直接继承 Runnables 创建新类，复杂且容易出错。

---

## ✅ 可配置 Runnables (Configurable Runnables)

* 高级功能，主要用于 LCEL 组合的大链 (chains)，以及 **LangServe 部署**。
* 提供两类方法：

  * `configurable_fields`：配置某个属性（如 ChatModel 的 `temperature`）。
  * `configurable_alternatives`：在多个 Runnables 之间切换（如不同的 ChatModel）。

---

 