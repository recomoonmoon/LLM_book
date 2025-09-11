
---
# 📘 LangChain Retrieval & RAG 笔记

## 📑 目录

* [1. Retrievers 检索器](#1-retrievers-检索器)

  * [1.1 前置知识](#11-前置知识)
  * [1.2 概述](#12-概述)
  * [1.3 接口与关键概念](#13-接口与关键概念)
  * [1.4 常见类型](#14-常见类型)
  * [1.5 高级检索模式](#15-高级检索模式)
* [2. Retrieval Augmented Generation (RAG)](#2-retrieval-augmented-generation-rag)

  * [2.1 概述](#21-概述)
  * [2.2 关键概念](#22-关键概念)
  * [2.3 工作流程](#23-工作流程)
  * [2.4 RAG 示例代码](#24-rag-示例代码)
  * [2.5 优势](#25-优势)
  * [2.6 延伸阅读](#26-延伸阅读)

---

## 1. Retrievers 检索器

### 1.1 前置知识

* **Vector Stores 向量存储**
* **Embeddings 向量化**
* **Text Splitters 文本切分**

### 1.2 概述

检索系统是现代 AI 应用（尤其是 **RAG**）的重要组成部分。
LangChain 提供了一个 **统一的检索接口**，兼容不同存储和查询方式。

* **输入**：自然语言查询（`str`）
* **输出**：文档列表（`Document` 对象）

### 1.3 接口与关键概念

**Retriever = 接收 query，返回相关 Document 列表**

* `page_content`: 文本内容（字符串）
* `metadata`: 文档元数据（id, 来源, 文件名等）

```python
docs = retriever.invoke("What is LangChain?")
```

👉 本质：Retriever 是一个 **Runnable**，可用 `invoke` 调用。

### 1.4 常见类型

1. **Search APIs**

   * 不存储文档，直接基于外部搜索接口（如 Amazon Kendra, Wikipedia Search）。

2. **关系型 / 图数据库**

   * 将自然语言转为 SQL / Cypher 查询。
   * 用于结构化数据检索。

3. **词法搜索（Lexical Search）**

   * 基于关键词匹配（BM25, TF-IDF, Elasticsearch）。

4. **向量存储（Vector Stores）**

   * 基于 Embedding 向量检索。
   * 典型写法：

     ```python
     retriever = vectorstore.as_retriever()
     ```

### 1.5 高级检索模式

#### (1) **Ensemble 检索器**

* 组合多个检索器（如 BM25 + 向量检索）
* 可加权求和或使用 **重排序（RRF, Re-ranking）**

```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_store_retriever],
    weights=[0.5, 0.5]
)
```

#### (2) **Source Document Retention**

* 保持索引后的 chunk 与原始文档的映射

* 防止模型丢失上下文

* **ParentDocument Retriever**

  * chunk 用于索引，但返回完整原文

* **MultiVector Retriever**

  * 为每个文档生成多个向量（如摘要、假设问答）

| 名称             | 索引类型        | 是否用 LLM | 适用场景               | 描述          |
| -------------- | ----------- | ------- | ------------------ | ----------- |
| ParentDocument | 向量存储 + 文档存储 | 否       | 文档分块检索但希望返回完整内容    | 按块索引，返回父文档  |
| MultiVector    | 向量存储 + 文档存储 | 可选      | 希望索引文档的额外信息（摘要、问题） | 多向量索引，更丰富检索 |

---

## 2. Retrieval Augmented Generation (RAG)

### 2.1 概述

RAG（检索增强生成）通过结合 **检索系统 + LLM**，解决模型依赖固定训练数据的问题。

**流程**：

1. 检索器获取相关文档
2. 将检索结果作为上下文传递给 LLM
3. LLM 基于检索信息生成答案

### 2.2 关键概念

* **检索系统**：从知识库中找到相关信息
* **外部知识注入**：将检索内容注入到 prompt

### 2.3 工作流程

1. 接收用户查询
2. 检索器返回相关文档
3. 整合文档到 prompt
4. LLM 基于上下文回答

### 2.4 RAG 示例代码

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 系统提示：指导模型使用检索结果
system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Keep the answer concise.
Context: {context}"""

# 用户问题
question = "What are the main components of an LLM-powered autonomous agent system?"

# 1. 检索文档
docs = retriever.invoke(question)

# 2. 整合文档
docs_text = "".join(d.page_content for d in docs)
system_prompt_fmt = system_prompt.format(context=docs_text)

# 3. 创建模型
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 4. 调用生成
response = model.invoke([
    SystemMessage(content=system_prompt_fmt),
    HumanMessage(content=question)
])

print(response.content)
```

---

### 2.5 优势

* **信息最新**：避免训练数据过时
* **领域知识注入**：支持专有数据库
* **减少幻觉**：回答基于真实文档
* **低成本扩展**：无需频繁微调模型

### 2.6 延伸阅读

* [Retrievers 文档](https://python.langchain.com/docs/modules/data_connection/retrievers/)
* [RAG 综述博客 - Cameron Wolfe](https://cameronrwolfe.substack.com/)
* LangChain RAG 教程、How-to、Freecodecamp RAG 课程

---

📌 总结：

* **Retriever = 输入 query，输出文档**
* **RAG = Retriever + LLM**
* **核心价值**：让 LLM 动态获取外部知识，降低幻觉，提升准确性

---
 