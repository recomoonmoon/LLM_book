from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
llm = Tongyi(model=model, temperature=0.7)
# 1. 定义 prompt，注意要有 MessagesPlaceholder 来插入历史

messages = [
    SystemMessage(content="你要扮演一个数学家庭教师，一步一步地教导数学"),
    HumanMessage(content="告诉我一加一等于几，并说出求解过程"),
]

for token in llm.stream(messages):
    print(token, end="|")

