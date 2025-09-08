import openai
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(dotenv_path="../record.env")
openai.api_key = os.environ["QWEN_API_KEY"]



