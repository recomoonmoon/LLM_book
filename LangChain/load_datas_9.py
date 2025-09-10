from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import langchain
from langchain_community.chat_models import ChatTongyi
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader



if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
#tongyi 是社区模型， 所以接口稍微有点小差异
#llm1 = ChatTongyi(model="qwen-turbo", temperature=0.0)

csv_loader = CSVLoader(
    file_path="./dataset/test.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"'
    },
    encoding='utf-8'
)

docs = csv_loader.load()
print(docs)
print(docs[0].page_content)
print(docs[0].metadata)

# 流式加载（节省内存）
for doc in csv_loader.lazy_load():
    print(doc.page_content[:50])