import langchain
import langsmith
import getpass
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.output_parsers import ResponseSchema, StructuredOutputParser,PydanticOutputParser,ListOutputParser,RetryOutputParser,OutputFixingParser,CommaSeparatedListOutputParser
from langchain_community.llms import Tongyi
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from langchain.chains import ConversationChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import *


if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
#tongyi 是社区模型， 所以接口稍微有点小差异
llm = Tongyi(model=model, temperature=0.7)


"""
test1 使用各类message，实现一个多轮对话的医生聊天机器人
"""

def test1():
    messages = []
    sys_message = SystemMessage(
        content="请你扮演一位医生的角色接待病人，和病人进行多轮对话，根据病人的情况，诊断症状，简洁地告诉他们需要什么药品，什么治疗")
    messages.append(sys_message)
    input_str = str(input("跳出聊天请输入“exit”\n"))
    while input_str != "exit":
        messages.append(HumanMessage(content=input_str))
        print("*" * 100)
        ai_response = llm.invoke(messages)
        messages.append(AIMessage(content=ai_response))
        print(ai_response)
        print("*" * 100)
        input_str = str(input("跳出聊天请输入“exit”\n"))

"""
test2 在test1基础上，实现对话的裁剪，控制上下文窗口的长度
"""
