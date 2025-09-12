
---

# LangChain Callbacks 使用指南

## 目录
1. [简介](#简介)
2. [回调事件（Callback Events）](#回调事件callback-events)
3. [回调处理器（Callback Handlers）](#回调处理器callback-handlers)
4. [回调传递方式](#回调传递方式)
5. [Python <=3.10 特殊情况](#python-310-特殊情况)
6. [示例代码](#示例代码)
7. [典型应用场景](#典型应用场景)
8. [注意事项](#注意事项)

---

## 简介
LangChain 提供了一个 **回调系统（Callbacks）**，允许在 LLM 或链条运行的不同阶段插入自定义逻辑。

常见用途：
- 日志记录
- 监控运行情况
- 流式输出（逐 token 打印生成结果）
- 调试和性能分析

使用方式：
- 通过 `callbacks` 参数传入自定义的 handler 对象
- handler 对象需实现相应回调方法

---

## 回调事件（Callback Events）

LangChain 内部定义了多种事件，每个事件对应一个回调方法：

| 阶段 | 触发时机 | 方法 |
|------|-----------|------|
| Chat model start | 聊天模型开始运行 | `on_chat_model_start` |
| LLM start | LLM 开始运行 | `on_llm_start` |
| LLM new token | LLM/Chat 模型输出新 token | `on_llm_new_token` |
| LLM ends | LLM/Chat 模型运行结束 | `on_llm_end` |
| LLM errors | LLM/Chat 出错 | `on_llm_error` |
| Chain start | Chain 开始运行 | `on_chain_start` |
| Chain end | Chain 结束运行 | `on_chain_end` |
| Chain error | Chain 出错 | `on_chain_error` |
| Tool start | Tool 开始执行 | `on_tool_start` |
| Tool end | Tool 执行结束 | `on_tool_end` |
| Tool error | Tool 出错 | `on_tool_error` |
| Agent action | Agent 执行动作 | `on_agent_action` |
| Agent finish | Agent 结束运行 | `on_agent_finish` |
| Retriever start | Retriever 开始执行 | `on_retriever_start` |
| Retriever end | Retriever 执行结束 | `on_retriever_end` |
| Retriever error | Retriever 出错 | `on_retriever_error` |
| Text | 任意文本执行 | `on_text` |
| Retry | 重试事件 | `on_retry` |

---

## 回调处理器（Callback Handlers）

Handler 是处理回调事件的对象，分为两类：

1. **同步 Handler (Sync)**
   - 继承 `BaseCallbackHandler`
2. **异步 Handler (Async)**
   - 继承 `AsyncCallbackHandler`

LangChain 会通过 `CallbackManager` 或 `AsyncCallbackManager` 调用注册的 handler 方法。

---

## 回调传递方式

### 1. 请求级回调（Request-time callbacks）
- 在调用时传入
- 会自动传递给子对象
- 示例：
```python
chain.invoke({"number": 25}, {"callbacks": [handler]})
````

### 2. 构造函数级回调（Constructor callbacks）

* 在对象初始化时传入
* **不会**传递给子对象
* 示例：

```python
chain = SomeChain(callbacks=[handler])
```

> ⚠️ 如果创建自定义 Chain 或 Runnable，需要手动传递 request-time callbacks 给子对象。

---

## Python <=3.10 特殊情况

* 对于 `RunnableLambda`、`RunnableGenerator` 或自定义 Tool，在 Python 3.10 及以下版本异步调用时：

  * LangChain 无法自动传播 callbacks
  * 需要手动将 callbacks 传递给子对象，否则事件不会触发

---

## 示例代码

```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import Tongyi

# 自定义 Handler
class PrintCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM 开始运行:", prompts)

    def on_llm_new_token(self, token, **kwargs):
        print("新 Token:", token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        print("\nLLM 运行结束:", response)

# 初始化模型并注册 callback
llm = Tongyi(model="qwen-turbo", api_url="https://api.example.com", callbacks=[PrintCallbackHandler()])

# 调用 LLM
resp = llm.invoke("写一首关于 LangChain 的五言诗")
print("\n最终输出:", resp)
```

> 如果 LLM 支持流式生成，需要设置 `streaming=True` 才能触发 `on_llm_new_token`。

---

## 典型应用场景

* **日志系统**：记录每次模型调用的输入和输出
* **流式展示**：实现类似 ChatGPT 的逐字输出
* **监控告警**：在工具或链条出错时触发警报
* **性能分析**：统计调用时间、token 使用量等

---

## 注意事项

1. 构造函数回调只作用于当前对象，不会自动继承到子对象。
2. 自定义异步 runnable 在 Python<=3.10 下需要手动传播 callbacks。
3. 流式 token 输出需要模型本身支持 streaming。
4. 对于工具调用（Tool calls），部分模型要求 `AIMessage` 后跟 `ToolMessage` 才能正常触发回调。


