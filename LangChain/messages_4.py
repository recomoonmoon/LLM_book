import langchain
import langsmith
import getpass
import os
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import *
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

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
def test2():

    messages = []
    messages.append(SystemMessage(content="你是一个专业医生，负责根据病人情况诊断病情和开处处方"))
    messages.append(HumanMessage(content="我肚子疼怎么办"))
    messages.append(AIMessage(content="你好，我是医生。你肚子痛多久了？是突然发作的还是慢慢开始的？疼痛的位置在哪里？是持续性的还是阵发性的？有没有其他症状，比如发烧、恶心、呕吐、腹泻或者便秘？"))
    messages.append(HumanMessage(content="十几天，持续的，有便秘"))

    print(messages)
    print("*"*100)
    messages = trim_messages(
        messages=messages,
        max_tokens=30,
        token_counter=count_tokens_approximately, #基于消息数量可以使用len
        strategy="last",
        allow_partial=False,
        end_on=("ai", "human"),
        start_on=("tool", "human", "sys"),
        include_system=True
    )
    print(messages)

