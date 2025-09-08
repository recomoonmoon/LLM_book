import langchain
import langsmith
import getpass
import os
import openai
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_community.llms import Tongyi

if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
llm = Tongyi(model=model, temperature=0.7)

"""
第一部分：LangChain Prompt 用法总结
=========================
Prompt 是 LLM 的输入，可以是字符串，也可以是多轮对话消息。
"""



"""
一、PromptTemplate 的用法（单轮字符串模板）
----------------------------------------
1. 构造 prompt，变量使用 {key} 填充
2. 有两种方式创建 PromptTemplate
   - PromptTemplate.from_template(template)
   - PromptTemplate(template=..., input_variables=[...])
3. 调用方式
   - .format(**vars)
   - 通过 chain 传参
"""
prompt = "---{disease}---有---{symptom}---症状，需要使用---{medicine}---药品进行治疗"
var_dict = {"disease": "糖尿病", "symptom": "尿血", "medicine": "格列美脲"}

prompt_template1 = PromptTemplate.from_template(prompt)
prompt_template2 = PromptTemplate(template=prompt, input_variables=["disease", "symptom", "medicine"])

print(prompt_template1.format(**var_dict))
print(prompt_template2.format(**var_dict))
print("*"*100)

# 串到 chain
chain = prompt_template1 | llm
print(chain.invoke(var_dict))
print("*"*100)


"""
二、FewShotPromptTemplate 的用法（示例提示）
-----------------------------------------
给模型几个输入输出示例，引导它学会格式。
"""
examples = [
    {"word": "cat", "translation": "猫"},
    {"word": "dog", "translation": "狗"}
]

example_prompt = PromptTemplate.from_template("英文: {word} -> 中文: {translation}")
fewshot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="英文: {word} -> 中文:",   # 留出用户输入
    input_variables=["word"]
)

print(fewshot_prompt.format(word="apple"))
print("*"*100)


"""
三、ChatPromptTemplate 的用法（多轮对话）
--------------------------------------
由 system / human / ai 等角色消息构成。
"""
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个医学助手。"),
    ("human", "病人患有{disease}，出现了{symptom}，应该如何治疗？")
])

print(chat_prompt.format(disease="糖尿病", symptom="尿血"))
print("*"*100)

# 串到 chain
chat_chain = chat_prompt | llm
print(chat_chain.invoke({"disease": "糖尿病", "symptom": "尿血"}))
print("*"*100)


"""
四、MessagePromptTemplate（细粒度控制单条消息）
---------------------------------------------
可以精确指定 System / Human / AI 消息。
"""
chat_prompt2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个翻译助手。"),
    HumanMessagePromptTemplate.from_template("请翻译这句话: {sentence}")
])

print(chat_prompt2.format(sentence="我今天很开心"))
print("*"*100)

"""
五、MessagesPlaceholder（动态插入历史对话）
-----------------------------------------
常用于多轮对话记忆，把历史聊天记录放进 prompt。
"""
chat_with_memory = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "请继续回答: {question}")
])

history = [
    {"role": "human", "content": "你好"},
    {"role": "ai", "content": "你好，我能帮你什么？"}
]

print(chat_with_memory.format(history=history, question="今天天气怎么样？"))
print("*"*100)
