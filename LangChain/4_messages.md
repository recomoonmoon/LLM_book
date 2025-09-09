

# LangChain Messages 教程笔记

## 目录
1. [基础概念](#基础概念)
   * 1.1 [消息类型](#消息类型)
   * 1.2 [常见接口](#常见接口)
2. [LangChain 消息对象](#LangChain-消息对象)
   * 2.1 [SystemMessage](#SystemMessage)
   * 2.2 [HumanMessage](#HumanMessage)
   * 2.3 [AIMessage](#AIMessage)
   * 2.4 [AIMessageChunk](#AIMessageChunk)
   * 2.5 [ToolMessage](#ToolMessage)
   * 2.6 [DeleteMessage](#DeleteMessage)
3. [消息裁剪 trim\_messages](#消息裁剪-trim_messages)
   * 3.1 [基于 Token 计数裁剪](#基于-Token-计数裁剪)
   * 3.2 [基于消息数量裁剪](#基于消息数量裁剪)
   * 3.3 [高级用法](#高级用法)
   * 3.4 [自定义 Token 计数器](#自定义-Token-计数器)
   * 3.5 [与 Runnable 结合使用](#与-Runnable-结合使用)
   * 3.6 [与 ChatMessageHistory 结合使用](#与-ChatMessageHistory-结合使用)
4. [参考链接与 API](#参考链接与-API)

---

## 基础概念

### 消息类型

LangChain 定义了统一的消息格式，屏蔽了不同模型提供商在消息格式上的差异。常见角色（Role）包括：

| Role                  | 描述                        |
| --------------------- | ------------------------- |
| **system**            | 指示模型如何行为或提供上下文（并非所有模型支持）。 |
| **user**              | 用户输入，通常是文本或其他交互输入。        |
| **assistant**         | 模型输出，可包含文本、工具调用等。         |
| **tool**              | 工具调用的结果，用于模型与外部数据交互。      |
| **function (legacy)** | 旧版 API，现已由 `tool` 替代。     |

### 常见接口

* **SystemMessage**：系统消息
* **HumanMessage**：用户消息
* **AIMessage**：模型消息
* **AIMessageChunk**：流式模型消息
* **ToolMessage**：工具调用消息

### Messages 裁剪

由于上下文窗口有限，需要对消息进行裁剪，以便在保证语义连贯的同时节省 Token。

---

## LangChain 消息对象

### SystemMessage

系统角色，用于设定对话的全局指令。

```python
from langchain_core.messages import SystemMessage
SystemMessage(content="You're a helpful assistant.")
```

---

### HumanMessage

对应 `user` 角色，表示用户输入。

```python
from langchain_core.messages import HumanMessage
model.invoke([HumanMessage(content="Hello, how are you?")])
```

---

### AIMessage

对应 `assistant` 角色，表示模型的回复。

```python
from langchain_core.messages import HumanMessage
ai_message = model.invoke([HumanMessage("Tell me a joke")])
print(ai_message)  # <-- AIMessage
```

常见属性：

* **content**：消息内容（字符串或内容块）
* **tool\_calls**：模型触发的工具调用
* **usage\_metadata**：Token 使用统计
* **id**：消息唯一标识符

---

### AIMessageChunk

用于流式输出，逐步返回模型的响应。

```python
for chunk in model.stream([HumanMessage("what color is the sky?")]):
    print(chunk)
```

支持通过 `+` 运算符合并为完整响应：

```python
ai_message = chunk1 + chunk2 + chunk3
```

---

### ToolMessage

工具调用的结果，附带 `tool_call_id` 和 `artifact` 字段。

```python
from langchain_core.messages import ToolMessage
ToolMessage(content="search results", tool_call_id="123")
```

---

### DeleteMessage

特殊消息类型，用于管理 `LangGraph` 中的历史记录，不对应任何角色。

---

## 消息裁剪 trim\_messages

### 基于 Token 计数裁剪

通过 `token_counter` 限制消息历史的 Token 数。

```python
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
)
from langchain_core.messages.utils import count_tokens_approximately

trimmed = trim_messages(
    messages,
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=45,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)
```

---

### 基于消息数量裁剪

通过 `len` 函数按消息数量裁剪。

```python
trimmed = trim_messages(
    messages,
    strategy="last",
    token_counter=len,
    max_tokens=5,
    include_system=True,
)
```

---

### 高级用法

允许部分保留（`allow_partial=True`）或删除 `SystemMessage`。

```python
trim_messages(messages, max_tokens=56, strategy="last", allow_partial=True)
```

---

### 自定义 Token 计数器

可基于 `tiktoken` 自定义计数器。

```python
import tiktoken
from typing import List
from langchain_core.messages import BaseMessage

def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def tiktoken_counter(messages: List[BaseMessage]) -> int:
    # custom token count logic
    return sum(str_token_counter(msg.content) for msg in messages)

trim_messages(messages, token_counter=tiktoken_counter, max_tokens=45)
```

---

### 与 Runnable 结合使用

`trim_messages` 可与 `ChatOpenAI` 组合成链式调用。

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
trimmer = trim_messages(token_counter=llm, max_tokens=45, strategy="last")
chain = trimmer | llm
chain.invoke(messages)
```

---

### 与 ChatMessageHistory 结合使用

用于管理长对话记录。

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chat_history = InMemoryChatMessageHistory(messages=messages[:-1])

chain_with_history = RunnableWithMessageHistory(chain, lambda _: chat_history)
chain_with_history.invoke([HumanMessage("what do you call a speechless parrot")])
```

---

## 参考链接与 API

* **LangChain API 文档**:

  * [SystemMessage](https://api.python.langchain.com/en/latest/messages/systemmessage.html)
  * [HumanMessage](https://api.python.langchain.com/en/latest/messages/humanmessage.html)
  * [AIMessage](https://api.python.langchain.com/en/latest/messages/aimessage.html)
  * [ToolMessage](https://api.python.langchain.com/en/latest/messages/toolmessage.html)
  * [trim\_messages](https://api.python.langchain.com/en/latest/messages/trim_messages.html)
  * [count\_tokens\_approximately](https://api.python.langchain.com/en/latest/messages/utils.html)

---

 