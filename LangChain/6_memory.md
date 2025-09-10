
---

# 1. Memory 概述

**知识点**

* \*\*Memory（记忆）\*\*是 AI Agent 的核心能力之一。
* 它分为 **短期记忆（short-term memory）** 和 **长期记忆（long-term memory）**：

  * 短期记忆：维护当前会话的上下文（thread 级别）。
  * 长期记忆：跨会话/跨线程存储用户或应用数据。

---

# 2. 短期记忆（Short-term memory）

**知识点**

* 短期记忆是**会话级别**的，用于维持当前对话的上下文。
* 它通常保存在 **LangGraph state** 中，并通过 **checkpointer** 存储，可以恢复对话。
* 短期记忆主要问题：LLM 的上下文窗口有限，太长的对话会导致模型“遗忘”或者计算成本过高。

**代码示例**（维护消息历史）

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 模拟一个短期记忆：消息历史
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hello, who are you?"),
    AIMessage("I am your assistant."),
    HumanMessage("Can you help me with LangGraph memory?")
]

# 简单打印历史
for m in messages:
    print(f"[{m.type}] {m.content}")
```

---

# 3. 管理短期记忆

**知识点**

* 长对话会超过模型上下文，需要**裁剪/过滤**历史消息。
* 常见方法：只保留最近 N 条消息、保留关键信息、提炼总结。

**代码示例**（裁剪消息）

```python
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

trimmed = trim_messages(
    messages,
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=50,   # 保留最多 50 tokens
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)

print("裁剪后的消息：")
for m in trimmed:
    print(f"[{m.type}] {m.content}")
```

---

# 4. 长期记忆（Long-term memory）

**知识点**

* **长期记忆**存储在\*\*命名空间（namespace）\*\*里，可以跨会话共享。
* 比如：记住用户偏好、历史事实、组织规则等。
* 存储方式：文档（document）形式，通常是 JSON。

**代码示例**（保存和检索长期记忆）

```python
from langgraph.store.memory import InMemoryStore

# 假设有一个 embedding 函数
def embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 2.0] * len(texts)]  # 假的向量

# 创建内存存储
store = InMemoryStore(index={"embed": embed, "dims": 2})

user_id = "user_123"
namespace = (user_id, "chat")

# 存储长期记忆
store.put(
    namespace,
    "preferences",
    {
        "language": "English",
        "style": "short, direct",
        "likes": ["Python", "LangChain"]
    }
)

# 获取指定记忆
item = store.get(namespace, "preferences")
print("用户偏好:", item.value)

# 搜索记忆
results = store.search(namespace, query="What language does user prefer?")
print("搜索结果:", results)
```

---

# 5. 记忆类型

**知识点**
LangGraph 借鉴心理学，把记忆分为三类：

* **语义记忆 (Semantic Memory)**：事实，例如“用户喜欢 Python”。
* **情景记忆 (Episodic Memory)**：经历/动作，例如“用户昨天上传了一个文件”。
* **程序记忆 (Procedural Memory)**：操作规则，例如“始终用简短回答”。

---

# 6. 语义记忆（Semantic Memory）

**知识点**

* 存储**事实性知识**，常用于个性化。
* 可以用两种方式组织：

  * **Profile**：单个文档，持续更新（如用户画像 JSON）。
  * **Collection**：多个文档集合，按需增加（容易扩展，但需要搜索）。

**代码示例**（Profile 方式）

```python
user_profile = {
    "name": "Alice",
    "preferred_language": "Python",
    "hobbies": ["AI", "Reading"]
}

# 更新用户画像
user_profile["hobbies"].append("Traveling")
print("用户画像:", user_profile)
```

---

# 7. 情景记忆（Episodic Memory）

**知识点**

* 存储**过去的事件/操作**，常用 few-shot 学习来复用过去经验。

**代码示例**（few-shot 示例）

```python
few_shot_examples = [
    {"input": "Add 2 and 3", "output": "5"},
    {"input": "Multiply 4 and 5", "output": "20"}
]

# 新问题
query = "Add 10 and 7"
print("提示中加入few-shot示例:", few_shot_examples, "当前问题:", query)
```

---

# 8. 程序记忆（Procedural Memory）

**知识点**

* 存储**规则/提示词**，可以通过“反思（reflection）”方式更新。
* 常见应用：**更新 system prompt** 来调整 Agent 行为。

**代码示例**（更新提示词）

```python
instructions = "You are a helpful assistant."
feedback = "Please answer more concisely."

# 更新后的 prompt
new_instructions = instructions + " " + feedback
print("新的 System Prompt:", new_instructions)
```

---

# 9. 写入记忆的两种方式

**知识点**

* **热路径（in the hot path）**：实时写入，立即生效（缺点：可能影响响应速度）。
* **后台写入（in the background）**：异步更新，不影响主流程（缺点：更新可能不及时）。

---

# 10. Memory 存储结构

**知识点**

* 记忆按 `namespace + key` 组织，类似文件夹+文件名。
* 支持跨 namespace 检索。

---

📌 总结：

* 短期记忆：会话内上下文，注意裁剪。
* 长期记忆：跨会话，通常 JSON 存储。
* 记忆类型：语义（facts）、情景（experiences）、程序（rules）。
* 更新方式：热路径（实时） vs 后台（异步）。

---

 