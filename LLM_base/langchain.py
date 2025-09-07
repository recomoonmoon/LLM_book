import langchain
import langsmith
import getpass
import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI

if load_dotenv("../record.env"):
    key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key





