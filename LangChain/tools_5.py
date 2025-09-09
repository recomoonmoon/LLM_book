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

if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
#tongyi 是社区模型， 所以接口稍微有点小差异
llm = ChatTongyi(model="qwen-turbo", temperature=0.7)


"""test1 创建一个求两数最小公约数的工具给llm使用"""

# 真正的 gcd 算法（普通 Python 函数）
def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

# 包装成 LangChain Tool
@tool
def gcd(num1: int, num2: int) -> int:
    """返回两个整数的最大公约数"""
    return _gcd(num1, num2)

llm_with_tools = llm.bind_tools([gcd])
# 用户问一个问题
messages = [HumanMessage(content="4823 和 3828823 的最大公约数是多少")]


# 第一步：模型可能决定调用工具
response = llm_with_tools.invoke(messages)
print("第一次回复:", response)

# 如果模型确实调用了 gcd，就执行工具
if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = gcd.invoke(tool_call["args"])
    tool_msg = ToolMessage(
        content=str(result),
        name=tool_call["name"],
        tool_call_id=tool_call["id"]
    )

    # 第二步：把结果交还给模型，让它生成自然语言答案
    final_response = llm_with_tools.invoke(messages + [response, tool_msg])
    print("最终回答:", final_response.content)
