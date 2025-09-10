 
---

# 📄 什么是 Document Loader？

在 LangChain 中，**文档加载器（Document Loader）** 用来从各种数据源读取数据，并转化为 **Document 对象**。

* **数据源**：Slack、Notion、Google Drive、CSV、PDF、本地文件夹、数据库、API……
* **输出**：统一的 **Document** 对象，包含：

  * `page_content`（文档内容，字符串）
  * `metadata`（元信息，如来源、时间戳、作者）

这样，无论数据来自哪里，后续的 **拆分（Splitter）、向量化（Embedding）、存储（Vector Store）** 都能直接处理。

---

# ⚙️ 接口规范

所有文档加载器都实现了 `BaseLoader` 接口。

核心方法：

* `.load()`
  一次性加载所有文档，返回 `List[Document]`
* `.lazy_load()`
  生成器模式，逐条加载文档，适合大数据量场景，避免内存爆炸。

---

# 🛠️ 基本用法

以 CSVLoader 为例：

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="example.csv",  # 具体数据源参数
    csv_args={
        "delimiter": ",",
        "quotechar": '"'
    }
)

# 一次性加载
docs = loader.load()
print(docs[0].page_content)
print(docs[0].metadata)

# 流式加载（节省内存）
for doc in loader.lazy_load():
    print(doc.page_content[:50])
```

输出的 `Document` 对象类似：

```python
Document(
    page_content="Name: Alice, Age: 25, City: New York",
    metadata={"source": "example.csv", "row": 1}
)
```

---

# 📚 常见 Loader 类型

LangChain 有几百种 loader，可以根据应用场景选择：

### 📂 文件类

* `TextLoader`：加载纯文本
* `CSVLoader`：加载 CSV
* `UnstructuredPDFLoader`：加载 PDF
* `PyPDFLoader`：更稳定的 PDF 解析
* `Docx2txtLoader`：加载 Word

### 🌐 API / SaaS 集成

* `SlackLoader`：Slack 消息
* `NotionLoader`：Notion 页面
* `GoogleDriveLoader`：谷歌云端硬盘文档
* `GitLoader`：Git 仓库

### 🗄️ 数据库类

* `MongoDBLoader`
* `SQLDatabaseLoader`

---

# 🚀 使用 `.lazy_load()` 的场景

如果你的数据量很大（比如几 GB 的 CSV，或成千上万篇 PDF），不要用 `.load()`，否则会一次性加载到内存。

推荐：

```python
for doc in loader.lazy_load():
    process(doc)  # 逐条处理，节省内存
```

---

# 🔗 下一步（和其他组件的关系）

1. **加载文档**：Document Loader (`.load()`)
2. **切分文档**：Text Splitter（把长文档分成 chunk）
3. **向量化**：Embedding Model
4. **存储**：Vector Store（如 FAISS、Pinecone、Weaviate）

这是典型的 **RAG 管道（检索增强生成）** 的第一步。

---

# ✅ 总结

* **Document Loader 作用**：把多种数据源转成统一的 Document 对象
* **接口**：`load()` 一次性，`lazy_load()` 流式加载
* **适配范围广**：文本、PDF、API、数据库等
* **配合使用**：通常和 Splitter → Embedding → Vector Store 结合

---

 