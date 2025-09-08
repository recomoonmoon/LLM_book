import langchain
import langsmith
import getpass
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.output_parsers import ResponseSchema, StructuredOutputParser,PydanticOutputParser,ListOutputParser,RetryOutputParser,OutputFixingParser
from langchain_community.llms import Tongyi
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

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
#print(chain.invoke(var_dict))
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
#print(chat_chain.invoke({"disease": "糖尿病", "symptom": "尿血"}))
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

"""
一、ResponseSchema（定义输出字段）
用途：定义希望模型输出的字段名和说明。
写法：为每个字段创建 ResponseSchema 对象。
示例：
"""
schema = ResponseSchema(name="disease", description="判断出的病人所患的疾病", dtype=str)
print(schema)

"""
二、StructuredOutputParser（结构化解析器）和 三
用途：告诉模型必须严格按指定格式输出，并自动解析成 Python 字典。
结合 ResponseSchema 使用：
StructuredOutputParser.from_response_schemas(...)
获取 format_instructions，拼进 Prompt，引导模型输出 JSON 格式。
示例：
"""
# 定义 schema
res_schemas = [
    ResponseSchema(name="disease", description="判断出的病人所患的疾病", type="string"),
    ResponseSchema(name="medicine", description="治疗该疾病所需的药品", type="string")
]
sp = StructuredOutputParser.from_response_schemas(res_schemas)
prompt = """我身体不舒服，有{symptom}症状，怎么办？请你根据症状判断疾病并给出治疗药品。请严格按照 JSON 格式输出：{standard}"""
standard = sp.get_format_instructions()
prompt_template = PromptTemplate(template=prompt, partial_variables={"standard": standard})
prompt = prompt_template.partial(format_instructions=sp.get_format_instructions())
chain = prompt | llm | sp
var_dict = {"symptom": "头晕，体温高，咳嗽，无力，流鼻涕"}
#print(chain.invoke(var_dict))

"""
四、解析模型返回结果
模型调用后，先得到原始字符串，再通过 output_parser.parse(...) 转换为结构化结果。
"""

chain = prompt | llm
output = chain.invoke(var_dict)
print(output)
print(sp.parse(output))

"""
五. PydanticOutputParser（带验证的 JSON 解析）
定义一个 Pydantic 模型，包含字段和校验规则
使用 PydanticOutputParser(pydantic_object=Model) 创建解析器
输出会强制转为 Python 对象，并做 类型验证
"""

class data_process(BaseModel):
    #不需要定义函数
    disease: str = Field(description="疾病名称")
    medicine: str = Field(description="治疗药物")

pydantic_sp = PydanticOutputParser(pydantic_object=data_process)
print(pydantic_sp.parse(output))

"""六. ListOutputParser / CommaSeparatedListOutputParser（解析列表）
ListOutputParser：解析 markdown 风格的列表
CommaSeparatedListOutputParser：解析逗号分隔字符串"""
