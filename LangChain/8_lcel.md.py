from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langgraph.store.memory import InMemoryStore

if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
#tongyi 是社区模型， 所以接口稍微有点小差异
llm1 = ChatTongyi(model="qwen-turbo", temperature=0.0)
llm2 = ChatTongyi(model="qwen-max", temperature=0.0)


"""并行调用多模型"""
prompt = ChatPromptTemplate.from_messages([("human", "回答这个问题: {question}")])
chain = RunnableParallel(
    {
        "qwen_turbo": prompt|llm1,
        "qwen-max": prompt|llm2
    }
)
#{'qwen_turbo': AIMessage(content='天空之所以是蓝色的，主要是因为大气中的一种物理现象叫做“瑞利散射”（Rayleigh scattering）。\n\n### 原理简述：\n\n太阳光由多种颜色的光组成，这些颜色对应不同的波长。其中，蓝光的波长较短（大约400-450纳米），而红光的波长较长（大约620-750纳米）。\n\n当太阳光进入地球的大气层时，光线会与大气中的气体分子（如氮、氧等）以及微小的颗粒发生相互作用。**蓝光比红光更容易被散射**，因为散射强度与波长的四次方成反比（即：波长越短，散射越强）。\n\n### 结果：\n\n- 蓝光在大气中被向各个方向散射，使得整个天空看起来是蓝色的。\n- 而红光等长波光则较少被散射，更多地沿直线传播，因此在日出或日落时，太阳光穿过更厚的大气层，蓝光几乎都被散射掉，剩下的红光和橙光让天空呈现出红色或橙色。\n\n### 为什么不是紫色？\n\n虽然紫光的波长比蓝光还短，理论上应该散射得更强，但人眼对紫光的敏感度较低，而且太阳光谱中蓝光比紫光更丰富，所以我们看到的是蓝色而不是紫色。\n\n---\n\n### 总结：\n\n天空是蓝色的，是因为太阳光中的蓝光在大气中被强烈散射，使得我们从各个方向看到的都是蓝色的光。这种现象称为“瑞利散射”。', additional_kwargs={}, response_metadata={'model_name': 'qwen-turbo', 'finish_reason': 'stop', 'request_id': '4e74f33e-43f1-4c25-9daa-b40d16b871e7', 'token_usage': {'input_tokens': 22, 'output_tokens': 354, 'total_tokens': 376, 'prompt_tokens_details': {'cached_tokens': 0}}}, id='run--651dff4e-49dd-4ca1-b78d-ee1c339271fc-0'), 'qwen-max': AIMessage(content='天空之所以呈现蓝色，主要是因为大气中的分子和微小颗粒对太阳光的散射作用。这个现象可以通过瑞利散射（Rayleigh scattering）来解释。当阳光进入地球的大气层时，它包含了多种颜色的光，这些颜色共同构成了我们看到的白色光。然而，不同颜色的光波长各不相同：蓝光的波长较短，而红光的波长较长。\n\n根据瑞利散射理论，在遇到空气中的气体分子等小微粒时，波长较短的光线（如蓝色和紫色）比波长长的光线（如红色和黄色）更容易被散射。这意味着当太阳光穿过地球的大气层时，其中的蓝光会被向四面八方大量散射开来。因此，当我们抬头望向没有直接对着太阳的方向时，就会看到更多的蓝色光线到达我们的眼睛，从而使天空呈现出蓝色。\n\n值得注意的是，实际上紫光的波长更短，应该比蓝光散射得更多。但人眼对蓝光更为敏感，并且太阳发出的蓝光强度也高于紫光，再加上日间地平线附近尘埃等因素的影响，使得最终我们感知到的天空主要为蓝色而非紫色。', additional_kwargs={}, response_metadata={'model_name': 'qwen-max', 'finish_reason': 'stop', 'request_id': '38fcf26d-d68c-4f7d-8767-7c72bd8562f7', 'token_usage': {'input_tokens': 18, 'output_tokens': 255, 'total_tokens': 273, 'prompt_tokens_details': {'cached_tokens': 0}}}, id='run--cf177b2f-cf07-4cbd-b5fa-72d84ad92b00-0')}
result = chain.invoke({"question": "为什么天空是蓝色的？"})
print(result)
