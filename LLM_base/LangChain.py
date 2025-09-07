import langchain
import langsmith
import getpass
import os
import openai
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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


"""
test1 
1.学会如何设置环境变量
2.学会调用模型（设置参数)
"""


model = "qwen-turbo"
temperature = 0.7
prompt_p = '''
{var1} * {var2} 等于多少？
'''
vardict = {"var1":5, "var2":7}
response = invoke_and_chain(model=model, temperature=temperature, prompt_p=prompt_p, var_dict=vardict)
print(response)


