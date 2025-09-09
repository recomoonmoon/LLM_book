---

# LangChain Tools 教程笔记

## 目录

1. [工具概述](#工具概述)
2. [核心概念](#核心概念)
3. [工具接口](#工具接口)
4. [使用 @tool 创建工具](#使用-tool-创建工具)
5. [直接使用工具](#直接使用工具)
6. [配置工具 Schema](#配置工具-schema)
7. [工具产物 (Artifacts)](#工具产物-artifacts)
8. [特殊类型注解](#特殊类型注解)

   * InjectedToolArg
   * RunnableConfig
   * InjectedState
   * InjectedStore
9. [工具产物 vs. 注入状态](#工具产物-vs-注入状态)
10. [最佳实践](#最佳实践)
11. [工具包 (Toolkits)](#工具包-toolkits)
12. [完整 Demo：让模型调用工具](#完整-demo让模型调用工具)
13. [相关资源](#相关资源)

---

## 工具概述

* **工具（Tool）** 是 **Python 函数 + Schema** 的封装。
* Schema 描述：

  * 名称
  * 描述
  * 参数结构
* 工具可以传给支持 **Tool Calling** 的聊天模型，由模型请求执行。

---

## 核心概念

* 工具是模型与外部函数交互的接口。
* 使用 `@tool` 装饰器能快速把一个函数包装成 Tool 对象。
* 工具可以返回文本，也可以返回数据（artifact）。

---

## 工具接口

继承自 **BaseTool**（Runnable 接口）。

**关键属性**

* `name`：工具名称
* `description`：工具说明
* `args`：参数 JSON Schema

**关键方法**

* `invoke(args)`：同步调用
* `ainvoke(args)`：异步调用

---

## 使用 @tool 创建工具

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

---

## 直接使用工具

```python
multiply.invoke({"a": 2, "b": 3})  # 输出 6
print(multiply.name)         # multiply
print(multiply.description)  # Multiply two numbers.
print(multiply.args)         # JSON schema
```

---

## 配置工具 Schema

`@tool` 装饰器支持自定义：

* 名称
* 描述
* 参数定义来源（类型注解 / docstring）

---

## 工具产物 (Artifacts)

工具可返回 (content, artifact) 二元组：

```python
@tool(response_format="content_and_artifact")
def some_tool() -> tuple[str, dict]:
    """Return message + raw data."""
    return "给模型的可读信息", {"raw": [1, 2, 3]}
```

* **content**：模型能理解的文本
* **artifact**：下游组件可用的原始对象

---

## 特殊类型注解

### InjectedToolArg

隐藏参数，由运行时注入：

```python
from langchain_core.tools import InjectedToolArg

@tool
def user_specific_tool(data: str, user_id: InjectedToolArg) -> str:
    return f"User {user_id} processed {data}"
```

### RunnableConfig

注入运行时配置：

```python
from langchain_core.runnables import RunnableConfig

@tool
async def tool_with_config(x: int, config: RunnableConfig):
    return f"{x} with config {config}"
```

### InjectedState

注入 LangGraph 的 **全局状态**。

### InjectedStore

注入 LangGraph 的 **存储对象**。

---

## 工具产物 vs. 注入状态

| 对比项  | 工具产物 (Artifacts) | 注入状态 (Injected State) |
| ---- | ---------------- | --------------------- |
| 目的   | 工具间数据传递          | 维护全局执行状态              |
| 范围   | 局部               | 全局                    |
| 生命周期 | 工具调用一次           | 跨多步、可保存               |
| 场景   | 临时共享结果           | 会话记忆、用户上下文            |

---

## 最佳实践

1. 工具命名清晰，文档完整。
2. 单一职责，避免复杂逻辑。
3. 返回结果尽量 JSON-friendly。
4. 使用支持 Tool Calling 的模型（如 GPT-4o, GPT-4-turbo）。

---

## 工具包 (Toolkits)

一组相关工具的集合：

```python
toolkit = ExampleToolkit(...)
tools = toolkit.get_tools()
```

---

## 完整 Demo：让模型调用工具

### 1. 定义工具

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

### 2. 绑定到模型

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([multiply])
```

### 3. 模型提出调用

```python
messages = [HumanMessage(content="What's 8 times 7?")]
response = llm_with_tools.invoke(messages)
print(response.tool_calls)
```

### 4. 执行工具 & 返回

```python
from langchain_core.messages import ToolMessage

tool_result = multiply.invoke({"a": 8, "b": 7})
tool_msg = ToolMessage(
    content=str(tool_result),
    name="multiply",
    tool_call_id=response.tool_calls[0]["id"]
)

final_response = llm_with_tools.invoke(messages + [response, tool_msg])
print(final_response.content)  # "The result of 8 times 7 is 56."
```

---
---

## 多工具调用示例

当我们绑定多个工具时，模型会根据语义自动选择调用哪个工具。

### 1. 定义多个工具

```python
from langchain_core.tools import tool

@tool
def gcd(a: int, b: int) -> int:
    """返回 a 和 b 的最大公约数"""
    while b:
        a, b = b, a % b
    return a

@tool
def pow(base: int, exp: int) -> int:
    """计算 base^exp"""
    return base ** exp

@tool
def compute_similarity(x: int, y: int) -> str:
    """比较两个数是否接近（差值/较大值 < 0.1 认为接近）"""
    ratio = abs(x - y) / max(x, y)
    return "接近" if ratio < 0.1 else "不接近"
```

### 2. 绑定到模型

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [compute_similarity, gcd, pow]
llm_with_tools = llm.bind_tools(tools)
```

### 3. 用户提问

```python
question = HumanMessage(content="18337834和124932接近吗？")
response = llm_with_tools.invoke([question])
print(response.tool_calls)
```

可能输出：

```python
[{'name': 'compute_similarity', 'args': {'x': 18337834, 'y': 124932}, 'id': 'call_1'}]
```

### 4. 执行工具并返回结果

```python
tool_call = response.tool_calls[0]
tool_name, tool_args = tool_call["name"], tool_call["args"]

tool_map = {t.name: t for t in tools}
tool_result = tool_map[tool_name].invoke(tool_args)

tool_msg = ToolMessage(
    content=str(tool_result),
    name=tool_name,
    tool_call_id=tool_call["id"]
)

final_response = llm_with_tools.invoke([question, response, tool_msg])
print(final_response.content)
```

输出示例：

```
18337834 和 124932 不接近，因为它们的差距远大于 10%。
```

---

## 常见疑难点记录

### 1. `Tongyi` 没有 `bind_tools`

社区模型 `Tongyi` 早期版本没有实现 `bind_tools`，需要升级到支持 **Tool Calling** 的版本，或者改用 `ChatOpenAI` / `ChatAnthropic`。

### 2. 工具递归调用报错

如果在工具函数内部直接调用自己（例如 `gcd` 递归时用了 `gcd(...)` 而不是普通函数），会导致 **Tool 对象递归调用**，引发 `int object has no attribute parent_run_id` 报错。
✅ 解决：把递归逻辑写在函数内部，不要再调用装饰后的 Tool 对象。

```python
@tool
def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
```

### 3. 如何把工具执行结果传回模型

必须封装为 `ToolMessage`，并保证 `tool_call_id` 与模型请求一致，否则模型不会识别。

```python
tool_msg = ToolMessage(
    content=str(result),
    name=tool_name,
    tool_call_id=tool_call["id"]
)
```

---

## 相关资源

* [@tool API 文档](https://api.python.langchain.com/en/latest/tools/tool.html)
* [自定义工具指南](https://python.langchain.com/docs/how_to/custom_tools)
* [运行时参数](https://python.langchain.com/docs/how_to/runtime_values_tools)
* [LangGraph 工具指南](https://python.langchain.com/docs/langgraph/tools)

---

 
 