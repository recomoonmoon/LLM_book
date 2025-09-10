from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import langchain
from langchain_community.chat_models import ChatTongyi
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter,Language

if load_dotenv("../record.env"):
    key = os.environ["QWEN_API_KEY"]
    url = os.environ["QWEN_URL"]
    os.environ["DASHSCOPE_API_KEY"] = key
model = "qwen-turbo"


"""test1 根据字符 和 token直接进行划分"""
# text = "".join(open().readlines())估计最快 但培养用框架的习惯还是这样写吧

loader = TextLoader("./dataset/test.txt", 'utf-8')
documents = loader.load()   # 返回 List[Document]
text = documents[0].page_content

def test1():
    # 根据字符数分割
    text_splitter = CharacterTextSplitter(
        chunk_size=50,  # 50字符
        chunk_overlap=0,
        separator="\n"  # 按换行或字符切
    )
    # 根据token分割 text_splitter = CharacterTextSplitter.from_tiktoken_encoder()
    chunks = text_splitter.split_text(text)
    # # 切分文本
    # # 打印结果
    for i, chunk in enumerate(chunks):
        print(f"--- chunk {i} ---")
        print(chunk)

"""test2 根据文档结构进行划分"""

def test2():
    loader = TextLoader("./1_prompt.md", 'utf-8')
    documents = loader.load()  # 返回 List[Document]
    text = documents[0].page_content

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=100,
        chunk_overlap=0
    )

    chunks = text_splitter.split_text(
        text
    )

    for i, chunk in enumerate(chunks):
        print(f"--- chunk {i} ---")
        print(chunk)

test2()