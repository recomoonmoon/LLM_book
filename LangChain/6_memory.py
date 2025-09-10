from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import langchain
import langsmith
import getpass
from langchain_community.chat_models import ChatTongyi
import os
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.store.memory import InMemoryStore

if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
#tongyi 是社区模型， 所以接口稍微有点小差异
llm = ChatTongyi(model="qwen-turbo", temperature=0.7)

"""
test1 什么是短期记忆？短期记忆就是上下文窗口，太大的上下文窗口会导致成本（输入token）高，影响最后对话的回答质量。
"""

# 模拟一个短期记忆：消息历史
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hello, who are you?"),
    AIMessage("I am your assistant."),
    HumanMessage("Can you help me with LangGraph memory?")
]
#print(llm.invoke(messages))

for m in messages:
    print(m)
print()

"""test2 短期记忆的管理方法，可以通过插入消息，MessagesPlaceholder替换，以及最常见的删除部分记忆（token_counter可以自己写）"""
trimed_messages = trim_messages(
    messages,
    max_tokens=100,
    token_counter=count_tokens_approximately)

for m in trimed_messages:
    print(m)


"""test3长期记忆存储在**命名空间（namespace）**里，可以跨会话共享。"""
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

