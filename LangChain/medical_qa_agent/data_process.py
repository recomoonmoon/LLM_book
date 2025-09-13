from bs4 import BeautifulSoup
import os
from langchain_community.chat_models import ChatTongyi

paths = [i for i in os.listdir("./data") if i.endswith(".html")]


def extract_text(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")

    # 找到目标元素
    elements = soup.find_all("div", class_="dus-bct-item")

    text = ""
    for el in elements:
        text += el.get_text(strip=True, separator="\n")
    if "简要说明书" in text:
        # print(f"==== {html_path} 提取结果 ====")
        # print(text)
        # print("=" * 80)
        return text
    else:
        return ""

col = []
for p in paths:
    text = extract_text(os.path.join("./data", p))
    if text:
        col.append(text)

for idx, text in enumerate(col):
    with open(f"./data/txt_medicine_intro/{idx}.txt", 'w+', encoding='utf-8') as f:
        f.write(text)
