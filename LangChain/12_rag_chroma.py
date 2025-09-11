import os
from dotenv import load_dotenv
from itertools import chain

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Tongyi

# =====================
# 环境变量加载
# =====================
load_dotenv("../record.env")
api_key = os.environ["QWEN_API_KEY"]
os.environ["DASHSCOPE_API_KEY"] = api_key

# 模型配置
llm_model = "qwen-turbo"
emb_model = "text-embedding-v1"

# =====================
# 初始化嵌入模型
# =====================
embedding_model = DashScopeEmbeddings(
    model=emb_model,
    dashscope_api_key=api_key
)

# =====================
# 加载所有 .md 文档
# =====================
docs = list(chain.from_iterable(
    TextLoader(f"./{fn}", encoding="utf-8").load()
    for fn in os.listdir("./") if fn.endswith(".md")
))

# =====================
# 文本切分
# =====================
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=500,
    chunk_overlap=50,
)

split_docs = splitter.split_documents(docs)

# =====================
# 构建 Chroma 向量库
# =====================
vector_store = Chroma.from_documents(split_docs, embedding_model)

# =====================
# 构建 RAG 检索 QA
# =====================
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=Tongyi(model=llm_model, temperature=0),
    retriever=retriever,
    return_source_documents=True
)

# =====================
# 测试问答
# =====================
query = "请问如何构造tools"
result = qa_chain.invoke({"query": query})

print("=== 答案 ===")
print(result["result"])

print("\n=== 来源文档 ===")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}")
