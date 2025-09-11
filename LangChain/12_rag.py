import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter,Language
from langchain_community.embeddings import DashScopeEmbeddings
from itertools import chain
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


load_dotenv("../record.env")
key = os.environ["QWEN_API_KEY"]
url = os.environ["QWEN_URL"]
os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"
emb_model = "text-embedding-v1"
ember = DashScopeEmbeddings(
    model = emb_model,
    dashscope_api_key=key
)

"""
很纳闷这个textloader的意义是什么，python自带open函数+“”，join能实现
"""
# 加载所有 md 文件为 Document
docs = list(chain.from_iterable(
    TextLoader(f"./{fn}", encoding="utf-8").load()
    for fn in os.listdir("./") if fn.endswith("t.md")
))

# 提取字符串内容
all_text = "\n".join(doc.page_content for doc in docs)

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=500,
    chunk_overlap=0
)

chunks = splitter.split_text(text="".join(all_text))

for idx, chunk in enumerate(chunks):
    print(f"--{idx}--")
    print(chunk)

embeddings = ember.embed_documents(chunks)
print(len(embeddings))
print(len(embeddings[0]))

