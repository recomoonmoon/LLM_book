 
---

# Embedding Models

（嵌入模型笔记）

## 📑 目录

1. [前置知识](#前置知识)
2. [概念概览](#概念概览)
3. [历史背景](#历史背景)
4. [接口与用法](#接口与用法)
5. [相似度度量](#相似度度量)
6. [进一步学习资源](#进一步学习资源)

---

## 前置知识

* **Documents**（文本数据）
* Embedding 模型主要针对 **文本**，尽管存在多模态 embedding，但目前 LangChain 不支持多模态。

---

## 概念概览

Embedding 模型的核心功能：

1. **Embed text as a vector**：将文本转换为定长向量表示（语义指纹）。
2. **Measure similarity**：使用数学方法比较向量，衡量语义相似度。

📌 价值：

* 不仅限于关键词匹配，而是基于语义理解实现检索、聚类、推荐等任务。

---

## 历史背景

* **2018 BERT**：Google 提出，首次使用 Transformer 结构做 embedding。 → 但不适合高效生成句子向量。
* **SBERT (Sentence-BERT)**：改造 BERT，使其高效生成语义句向量，支持余弦相似度等快速比较，大幅降低计算成本。
* **现在**：embedding 模型生态多元，研究者常参考 **MTEB (Massive Text Embedding Benchmark)** 来比较模型效果。

---

## 接口与用法

LangChain 提供统一接口，核心方法有：

* `embed_documents`：对多个文本生成向量（批量处理）。
* `embed_query`：对单个查询生成向量。

示例：

```python
from langchain_openai import OpenAIEmbeddings

# 初始化模型
embeddings_model = OpenAIEmbeddings()

# 文档批量 embedding
embeddings = embeddings_model.embed_documents([
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])

print(len(embeddings), len(embeddings[0]))  # (5, 1536)

# 查询 embedding
query_embedding = embeddings_model.embed_query("What is the meaning of life?")
```

📌 注意：部分 embedding 服务对 **查询** 和 **文档** 使用不同策略。

---

## 相似度度量

Embedding 向量位于高维语义空间，常见度量方法：

1. **Cosine Similarity**：角度相似性，最常用。
2. **Euclidean Distance**：欧式距离，直线距离。
3. **Dot Product**：点积，反映投影关系。

示例（余弦相似度）：

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_embedding, embeddings[0])
print("Cosine Similarity:", similarity)
```

---

## 进一步学习资源

* [BERT 原始论文](https://arxiv.org/abs/1810.04805)
* Cameron Wolfe 的 embedding 模型综述
* [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---

 