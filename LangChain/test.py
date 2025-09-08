from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os
if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
llm = Tongyi(model=model, temperature=0.7)
# 1. 定义 prompt，注意要有 MessagesPlaceholder 来插入历史
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])
# 3. 链接 prompt -> llm
chain = prompt | llm
# 4. 定义存储历史的函数
store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
# 5. 包装为带历史的 runnable
with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)
# 6. 调用
config = {"configurable": {"session_id": "user123"}}
resp1 = with_history.invoke({"input": "你好，我是小明"}, config=config)
print(resp1)
resp2 = with_history.invoke({"input": "我刚才说我是谁？"}, config=config)
print(resp2)
