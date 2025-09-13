import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

def download_htmls(url_file, save_dir):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 逐行读取 urls.txt
    with open(url_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    # 设置 Selenium Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无界面模式（如果你要看页面加载，可以注释掉）
    service = Service()  # 如果有 chromedriver.exe 路径，填在 Service("chromedriver路径")

    driver = webdriver.Chrome(service=service, options=chrome_options)

    for idx, url in enumerate(urls, start=1):
        try:
            print(f"正在下载 {url} ...")
            driver.get(url)
            time.sleep(1)  # 等待页面加载

            html_content = driver.page_source  # 获取完整 HTML

            # 文件名：page_1.html, page_2.html...
            filename = f"page_{idx}.html"
            filepath = os.path.join(save_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f_out:
                f_out.write(html_content)

            print(f"保存到 {filepath}")

        except Exception as e:
            print(f"下载失败 {url}: {e}")

    driver.quit()


# 使用示例
download_htmls("./data/urls.txt", "./data/")
