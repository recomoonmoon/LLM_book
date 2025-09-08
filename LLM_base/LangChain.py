import langchain
import langsmith
import getpass
import os
import openai
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.memory import ConversationBufferMemory


if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
    #client = Tongyi(model="qwen-turbo", temperature=0.7)

def invoke_and_chain(model, temperature, prompt_p, var_dict):
    client = Tongyi(model=model, temperature=temperature)
    prompt = PromptTemplate.from_template(prompt_p)
    chain = prompt | client
    return chain.invoke(var_dict)


# 模型设置
model = "qwen-turbo"
temperature = 0.7

"""
test1 
1.学会如何设置环境变量
2.学会调用模型（设置参数)
"""

"""
model = "qwen-turbo"
temperature = 0.7
prompt_p = '''
{var1} * {var2} 等于多少？
'''
vardict = {"var1":5, "var2":7}
response = test1(model=model, temperature=temperature, prompt_p=prompt_p, var_dict=vardict)
print(response)
"""

"""《--------------------------------------------------------------------------------------------》"""

"""
test 2
在项目基础上，增加输出的解析
"""
def test1(model, temperature, prompt_p, var_dict, parser):
    client = Tongyi(model=model, temperature=temperature)
    prompt = PromptTemplate.from_template(prompt_p)
    chain = prompt | client | parser
    return chain.invoke(var_dict)


# 定义输出模式
function_schema1 = ResponseSchema(
    name="乘数的字典",
    description="两数相乘里两个数字，以字典形式，格式为 {'a': 第一个数, 'b': 第二个数, '答案': 结果}"
)
# 构造 parser
parser = StructuredOutputParser.from_response_schemas([function_schema1])

def test2(model, temperature, function_schema1, parser):
    # 获取格式要求
    format_instructions = parser.get_format_instructions()
    # prompt 加入格式要求
    prompt_p = '''
    请回答以下问题，并严格按照指定格式输出。
    问题: {var1} * {var2} 等于多少？
    {format_instructions}
    '''
    prompt = PromptTemplate.from_template(prompt_p)
    # 构建 chain
    client = Tongyi(model=model, temperature=temperature)
    chain = prompt | client | parser
    # 输入
    vardict = {"var1": 5, "var2": 7, "format_instructions": format_instructions}
    response = chain.invoke(vardict)
    print("解析后的结果:", response)

