
---

## 🌐 什么是多模态 (Multimodality)

多模态指的是能够处理 **不同形式的数据**：

* 文本 (text)
* 图像 (image)
* 音频 (audio)
* 视频 (video)
* 文件 (file, e.g., PDF, Word)

LangChain 允许你把这些不同类型的数据作为输入，传给模型，或者让模型输出多模态的结果（比如 AI 回复里直接生成图片或音频）。

---

## 🧩 多模态在不同模块中的应用

* **Chat Models**
  聊天模型可以接受多模态输入和输出（比如传图片进去，模型描述图片；或者模型生成音频/图片作为输出）。

* **Embedding Models**
  把不同模态的数据（文本、图像、音频）转成向量表示，用于检索或对比。

* **Vector Stores**
  存储和检索多模态数据的向量表示。

---

## 💬 多模态在 Chat Models 中的使用

LangChain 提供了统一的 **消息格式**（`HumanMessage` 等），我们只需要在 `content` 里标明类型（`text` / `image` / `file` 等）即可。

### 1. 文本 + 图片 URL 输入

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "请描述这张图片里的天气情况:"},
        {
            "type": "image",
            "source_type": "url",
            "url": "https://example.com/weather.jpg",
        },
    ],
)

response = model.invoke([message])
print(response.content)
```

👉 模型会先读文字提示，再去分析图片。

---

### 2. 文本 + 内嵌图片（Base64）

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "请描述这张图片里的天气情况:"},
        {
            "type": "image",
            "source_type": "base64",
            "data": "<base64字符串>",
            "mime_type": "image/jpeg",
        },
    ],
)

response = model.invoke([message])
```

这种方式适合你手里已经有图片二进制数据，不方便上传 URL 时。

---

### 3. 文本 + 文件（比如 PDF）

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "总结这份文件的主要内容:"},
        {
            "type": "file",
            "source_type": "base64",
            "data": "<pdf的base64字符串>",
            "mime_type": "application/pdf",
        },
    ],
)

response = model.invoke([message])
```

---

## 📤 多模态输出

有些模型（比如 OpenAI, Gemini）能输出音频或图片。比如：

* 让 OpenAI 生成语音
* 让 Gemini 生成一张图片

这些输出会以 **AIMessage** 的形式返回，其中 `content` 里会包含 `{"type": "image", "data": ...}` 或音频数据。

---

## 🛠 工具 (Tools) 与多模态

目前工具调用本身不能直接传图片/音频文件 **内容**，但你可以：

* 传 URL 给工具
* 工具内部下载并处理图片/音频

比如：

```python
from langchain_core.tools import tool

@tool
def download_and_analyze_image(url: str) -> str:
    """下载并分析一张图片。"""
    # 这里写下载逻辑，比如requests.get(url)
    return f"图片 {url} 已被分析。"

tools = [download_and_analyze_image]
llm_with_tools = llm.bind_tools(tools)
```

---

## 🧠 Embedding 和 Vector Store

* 目前 LangChain 的 **embedding 接口只支持文本**。
* 未来会扩展到图像、音频、视频。
* 多模态检索的想法是：同一个向量空间里存不同模态的数据，支持跨模态搜索（例如用文字检索图片）。

---

⚡总结：

1. **输入**：通过 `HumanMessage.content` 传 `text`、`image`、`file` 等。
2. **输出**：部分模型可以返回图片或音频。
3. **工具**：可以通过 URL 间接处理多模态数据。
4. **Embedding/Vector Store**：现在只支持文本，未来会扩展。

---
 