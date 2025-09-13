import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.agents import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


load_dotenv("../../record.env")
api_key = os.environ["QWEN_API_KEY"]
api_url = os.environ["QWEN_URL"]
os.environ["DASHSCOPE_API_KEY"] = api_key
llm_model = "qwen-turbo"
embedding_model = "text-embedding-v1"


# ========== Step 1. 加载数据 ==========
data = json.load(open("./data/medicine_info.json", "r", encoding="utf-8"))

# ========== Step 2. 初始化 LLM & Embeddings ==========
llm = ChatTongyi(model="qwen-plus")  # 大模型
embedding = DashScopeEmbeddings(model="text-embedding-v1")  # 向量模型

# ========== Step 3. 构造向量库 ==========
docs = []
for item in data:
    metadata = {"药物": item["药物"], "靶点": item["靶点"]}
    content = json.dumps(item, ensure_ascii=False)
    docs.append(Document(page_content=content, metadata=metadata))

vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ========== Step 4. 定义 Tool ==========
# 定义输出 schema
class DrugEntities(BaseModel):
    药物: str = Field(description="药品名称，如果没有则填 '无'")
    靶点: str = Field(description="靶点名称，如果没有则填 '无'")

parser = JsonOutputParser(pydantic_object=DrugEntities)

@tool
def retrieve_medicine_data(query: str) -> str:
    """根据用户输入的问题，抽取药品名称或靶点实体，并从向量库检索相关药品信息"""

    # ===== 第一步：实体识别 =====
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个医学实体识别助手。"),
        ("human", """从下面的文本中提取药品名称和靶点。
如果没有提到，填 '无'。

文本: {query}

输出 JSON，格式严格符合：
{format_instructions}
""")
    ]).partial(format_instructions=parser.get_format_instructions())

    entity_resp = llm.invoke(prompt.format_messages(query=query)).content
    #print("识别原始输出：", entity_resp)

    try:
        entities = parser.parse(entity_resp)
    except Exception as e:
        print("解析失败:", e)
        entities = DrugEntities(药物="无", 靶点="无")

    # ===== 第二步：构造检索 query =====
    search_query = ""
    if entities["药物"] != "无":
        search_query += entities["药物"] + " "
    if entities["靶点"] != "无":
        search_query += entities["靶点"]

    if not search_query.strip():
        return "未检测到药品名称或靶点，不调用检索。"

    # ===== 第三步：向量检索 =====
    docs = retriever.invoke(search_query)
    if not docs:
        return f"没有找到与 {search_query} 相关的药品信息。"

    return "\n".join([doc.page_content for doc in docs])

# ========== Step 5. 抽象一个 Agent Runner ==========
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

class SimpleAgentRunner:
    def __init__(self, llm, tools, max_history: int = 10, system_prompt: str = None):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.llm_with_tools = llm.bind_tools(tools)
        self.max_history = max_history
        self.history = []

        # 如果有 system prompt，保存到 history 开头
        if system_prompt:
            self.history.append(SystemMessage(content=system_prompt))

    def _trim_history(self):
        """只保留 system + 最近 max_history 条消息"""
        system_msgs = [m for m in self.history if isinstance(m, SystemMessage)]
        other_msgs = [m for m in self.history if not isinstance(m, SystemMessage)]

        # 裁剪：保留最后 max_history 条
        if len(other_msgs) > self.max_history:
            other_msgs = other_msgs[-self.max_history:]

        self.history = system_msgs + other_msgs

    def chat(self, user_query: str) -> str:
        # 1. 添加用户消息
        self.history.append(HumanMessage(content=user_query))

        # 2. LLM 初次输出
        response = self.llm_with_tools.invoke(self.history)

        # 如果调用工具
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool_result = self.tools[tool_name].invoke(tool_args)

            # 工具消息
            tool_msg = ToolMessage(
                content=str(tool_result),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )

            self.history.append(response)  # 模型的中间输出
            self.history.append(tool_msg)  # 工具结果

            # 第二次调用模型
            final_response = self.llm_with_tools.invoke(self.history)
            self.history.append(final_response)

            # 对话裁剪
            self._trim_history()

            return final_response.content
        else:
            # 直接自然语言回复
            self.history.append(response)
            self._trim_history()
            return response.content


# ========== Step 6. Demo ==========
if __name__ == "__main__":
    agent = SimpleAgentRunner(llm, [retrieve_medicine_data])
    print("=== 多轮对话 Demo ===")
    while True:
        query = input("\n你: ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        answer = agent.chat(query)
        print("助手:", answer)
