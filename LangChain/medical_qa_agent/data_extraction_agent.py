import os
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatTongyi
from langchain_community.chat_models import ChatTongyi
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv



load_dotenv("../../record.env")
api_key = os.environ["QWEN_API_KEY"]
api_url = os.environ["QWEN_URL"]
os.environ["DASHSCOPE_API_KEY"] = api_key
llm_model = "qwen-turbo"
embedding_model = "text-embedding-v1"


# ========================
# 1. 定义 Pydantic 数据模型
# ========================
class MedicineInfo(BaseModel):
    药物: str = Field(..., description="药物名称")
    价格参考: Optional[str] = Field(None, description="药品价格参考")
    中国上市情况: Optional[str] = Field(None, description="中国上市情况")
    靶点: Optional[str] = Field(None, description="药物靶点")
    治疗: Optional[str] = Field(None, description="适应症/治疗")
    参考用法用量: Optional[str] = Field(None, description="推荐用法用量")
    不良反应: Optional[str] = Field(None, description="主要不良反应")


# ========================
# 2. 构造输出 parser
# ========================
parser = PydanticOutputParser(pydantic_object=MedicineInfo)

# ========================
# 3. 构造 PromptTemplate
# ========================
template = """
你是一个药品说明书数据抽取助手。
下面是一个药品说明书文本（可能包含一些噪声或无关信息）：

{text}

请根据文本提取以下信息，生成 JSON：
药物、价格参考、中国上市情况、靶点、治疗、参考用法用量、不良反应

要求：
- 输出严格符合 JSON 格式
- 如果某个字段在文本中找不到，可以置为 null
- 使用中文 key

返回结果：
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ========================
# 4. 初始化 LLM
# ========================
llm = ChatTongyi(
    model_name="qwen-max",  # 或者 "gpt-4" / "gpt-3.5-turbo"
    temperature=0
)

# ========================
# 5. 遍历文本，调用 LLM
# ========================
text_files = [ "./data/txt_medicine_intro/" + f
              for f in os.listdir("./data/txt_medicine_intro")
              if f.endswith(".txt")]

results = []

for path in text_files:
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(text)
    input_prompt = prompt.format(text=text)

    # 调用 LLM
    output = llm.invoke([HumanMessage(content=input_prompt)])

    # 使用 parser 解析输出
    try:
        parsed = parser.parse(output.content)
        print(parsed)
        results.append(parsed.model_dump())
    except Exception as e:
        print(f"{path} 解析失败: {e}")

# ========================
# 6. 保存结果
# ========================
import json

with open("./data/medicine_info.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("数据抽取完成，保存到 ./data/medicine_info.json")
