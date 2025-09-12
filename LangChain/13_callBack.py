from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import Tongyi
import os
from dotenv import load_dotenv

# ========= 配置环境 =========
load_dotenv("../record.env")
api_key = os.environ["QWEN_API_KEY"]
api_url = os.environ["QWEN_URL"]
os.environ["DASHSCOPE_API_KEY"] = api_key
llm_model = "qwen-turbo"

# 自定义 Handler
class PrintCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM 开始运行:", prompts)

    def on_llm_new_token(self, token, **kwargs):
        print("新 Token:", token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        print("\nLLM 运行结束:", response)

# 初始化模型，注册 callback
llm = Tongyi(model = llm_model, api_url = api_url, callbacks=[PrintCallbackHandler()], streaming=True)
# 调用
resp = llm.invoke("写一首关于 LangChain 的五言诗")

print("\n最终输出:", resp)
