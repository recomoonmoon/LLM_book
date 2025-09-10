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

"""test2 使用llm调用工具完成任务"""
def test2():
    llm_with_tools = llm.bind_tools([gcd])
    # 用户问一个问题
    messages = [HumanMessage(content="4823 和 3828823 的最大公约数是多少")]
    # 第一步：模型可能决定调用工具
    response = llm_with_tools.invoke(messages)
    print("第一次回复:", response)
    #第一次回复: content='' additional_kwargs={'tool_calls': [{'function': {'name': 'gcd', 'arguments': '{"num1": 4823, "num2": 3828823}'}, 'index': 0, 'id': 'call_8904f4b397774cebab13bf', 'type': 'function'}]} response_metadata={'model_name': 'qwen-turbo', 'finish_reason': 'tool_calls', 'request_id': '7d0d0f9a-33ab-4a48-aa0c-41e8cd0c43bd', 'token_usage': {'input_tokens': 184, 'output_tokens': 35, 'total_tokens': 219, 'prompt_tokens_details': {'cached_tokens': 0}}} id='run--7e628998-3771-40a1-8ccd-59952a49feec-0' tool_calls=[{'name': 'gcd', 'args': {'num1': 4823, 'num2': 3828823}, 'id': 'call_8904f4b397774cebab13bf', 'type': 'tool_call'}]
    # 如果模型确实调用了 gcd，就执行工具
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(response.tool_calls)
        result = gcd.invoke(tool_call["args"])
        #[{'name': 'gcd', 'args': {'num1': 4823, 'num2': 3828823}, 'id': 'call_8904f4b397774cebab13bf', 'type': 'tool_call'}]
        tool_msg = ToolMessage(
            content=str(result),
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        # 第二步：把结果交还给模型，让它生成自然语言答案
        final_response = llm_with_tools.invoke(messages + [response, tool_msg])
        print("最终回答:", final_response.content)
test2()

"""test3 创建多个工具，让llm在多个工具间选择解决问题"""

@tool
def pow(num1:int, num2:int) -> int:
    """返回num1的num2次方"""
    return num1**num2

@tool
def compute_similarity(num1: int, num2: int) -> float:
    """计算两个数字的相似度，相似度最大为1，最小为0
    规则：计算num1在num2中最长公共子串的长度 / num1的长度
    """
    num1 = str(num1)
    num2 = str(num2)
    if len(num1) > len(num2):
        num1, num2 = num2, num1
    max_match = 0
    for i in range(len(num1)):
        for j in range(i + 1, len(num1) + 1):
            sub = num1[i:j]
            if sub in num2:
                max_match = max(max_match, len(sub))
    return max_match / len(num1)

from langchain_core.messages import HumanMessage, ToolMessage

def test3():
    # 用户问题
    question = HumanMessage(content="18337834和124932接近吗？")

    # 工具列表
    tools = [compute_similarity, gcd, pow]
    llm_with_tools = llm.bind_tools(tools)

    # 模型初次输出
    response = llm_with_tools.invoke([question])
    print("LLM raw response:", response)

    # 如果模型调用了工具
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"模型选择调用工具: {tool_name}, 参数: {tool_args}")

        # 执行对应工具
        tool_map = {t.name: t for t in tools}
        tool_result = tool_map[tool_name].invoke(tool_args)
        print("工具执行结果:", tool_result)

        # 包装为 ToolMessage
        tool_msg = ToolMessage(
            content=str(tool_result),
            name=tool_name,
            tool_call_id=tool_call["id"]
        )

        # 把结果交还给模型，生成最终回答
        final_response = llm_with_tools.invoke([question, response, tool_msg])
        print("最终回答:", final_response.content)
    else:
        print("模型没有调用工具，直接回答:", response.content)

# 运行测试
test3()
