---

# 📑 文本切分器（Text Splitters）笔记

## 一、前置知识

* **Documents**：需要处理的文档对象。
* **Tokenization**：分词或分 token，是模型输入的基本单位。

## 二、概述

在很多应用中，**文档切分**是关键的预处理步骤。它的核心是将大文本切分为较小的块（chunk）。
好处包括：

* 统一处理不同长度的文档。
* 避免超过模型的最大输入限制。
* 提升表示质量（嵌入/向量化更精准）。
* 改进检索的精度（更细粒度的匹配）。
* 节省计算资源。

一句话总结：**切分让文档更小、更清晰、更适合模型处理。**

---

## 三、为什么要切分文档？

1. **应对文档长度不一**：真实世界中文档大小差异很大，切分后能统一处理。
2. **突破模型输入限制**：Embedding 模型或大模型都有输入长度上限。
3. **提高表示质量**：过长文本 embedding 会稀释语义，切分后每块更聚焦。
4. **提升检索精度**：支持段落级别的匹配，而不是整篇匹配。
5. **优化计算**：小块文本更省内存，也便于并行。

---

## 四、切分方法

### 1. **基于长度的切分**

* **思路**：按固定大小拆分（字符数 / token 数）。
* **优点**：

  * 简单直接
  * 块大小一致
  * 适配不同模型
* **常见方式**：

  * **Token-based**：按 token 数切分（适合 LLM）
  * **Character-based**：按字符数切分（通用）
* **例子**：

  ```python
  from langchain_text_splitters import CharacterTextSplitter

  text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
      encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
  )
  texts = text_splitter.split_text(document)
  ```

---

### 2. **基于文本结构的切分**

* **思路**：利用自然层级结构（段落 → 句子 → 单词）逐级切分。
* **实现**：`RecursiveCharacterTextSplitter`

  * 优先保留较大单元（段落）
  * 超过限制就往下切分（句子/单词）
* **例子**：

  ```python
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
  texts = text_splitter.split_text(document)
  ```

---

### 3. **基于文档结构的切分**

* **思路**：利用文件的结构特性来切分。
* **适用场景**：

  * Markdown → 按标题 `#`、`##` 切
  * HTML → 按标签切
  * JSON → 按对象 / 数组切
  * 代码 → 按函数、类、逻辑块切
* **优点**：

  * 保留文档原有逻辑
  * 上下文更连贯
  * 对检索和总结更有效

---

### 4. **基于语义的切分**

* **思路**：直接基于文本含义，而非表面结构。
* **方法**（例：滑动窗口+embedding 差异检测）：

  * 取一组句子 → 生成 embedding
  * 移动窗口取下一组句子 → 生成 embedding
  * 比较 embedding，相差大时作为切分点
* **优点**：

  * 语义连贯性更好
  * 下游任务表现更优（检索/总结）
* **缺点**：

  * 计算更复杂

---

## 五、总结

| 方法     | 特点                   | 优点      | 缺点          | 适用场景     |
| ------ | -------------------- | ------- | ----------- | -------- |
| 长度切分   | 固定大小                 | 简单统一    | 可能破坏语义      | 通用场景     |
| 文本结构切分 | 按段落/句子/词             | 保持自然语言流 | 部分长段落仍可能被打散 | 普通文档     |
| 文档结构切分 | 按 Markdown/HTML/JSON | 逻辑清晰    | 依赖文档格式      | 结构化文档    |
| 语义切分   | 基于 embedding         | 最语义连贯   | 成本高         | 高质量检索/总结 |

---
 