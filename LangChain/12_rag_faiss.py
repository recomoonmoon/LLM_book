import os
from itertools import chain
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Tongyi


# ========= 配置环境 =========
load_dotenv("../record.env")
api_key = os.environ["QWEN_API_KEY"]
api_url = os.environ["QWEN_URL"]
os.environ["DASHSCOPE_API_KEY"] = api_key
llm_model = "qwen-turbo"
embedding_model = "text-embedding-v1"


# ========= 初始化模型 =========
embedding_model = DashScopeEmbeddings(
    model=embedding_model,
    dashscope_api_key=api_key
)

llm = Tongyi(model=llm_model)


# ========= 加载文档 =========
def load_markdown_docs(folder_path="./", suffix=".md"):
    """加载指定目录下的 Markdown 文件"""
    docs = list(chain.from_iterable(
        TextLoader(os.path.join(folder_path, fn), encoding="utf-8").load()
        for fn in os.listdir(folder_path) if fn.endswith(suffix)
    ))
    return docs


documents = load_markdown_docs()


# ========= 文本切分 =========
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=500,
    chunk_overlap=50
)
split_docs = splitter.split_documents(documents)


# ========= 向量化存储 =========
vector_store = FAISS.from_documents(split_docs, embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# ========= 构建 RAG 问答链 =========
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# ========= 示例查询 =========
query = "请问langchain的tools如何创建"
result = qa_chain({"query": query})

print("回答：", result["result"])
print("\n引用来源：")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}")
