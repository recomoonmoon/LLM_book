---

学习路线笔记：从 PyTorch 到大语言模型

这是一个database组程序员学习 **LLM（Large Language Models）** 的学习记录与路线图。每一个章节内都有对应的篇章的笔记，对从0开始的同学友好，也适合有基础的程序员挑着看。

---

## 学习路线

### 1. PyTorch 基础

* **学习资料：**

  * [PyTorch 中文文档速通](https://github.com/chenyuntc/pytorch-book/blob/master)
  * [B站 PyTorch 视频教程](https://www.bilibili.com/video/BV1hE411t7RN/)

* **阶段目标：**
  掌握 PyTorch 常用模块与基本理论，能够独立实现经典 CNN 模型 **ResNet**。

* **成果：**

  * [代码与文档（ResNet）](https://github.com/recomoonmoon/LLM_learning_book/blob/master/ResNet/)

---

### 2. Transformer 基础

* **学习重点：**

  * 理解 Transformer 架构：位置编码、多头注意力、残差连接、LayerNorm 等
  * 掌握 Encoder-Decoder 整体流程

* **阶段目标：**
  能够独立复现 Transformer 架构模型。

* **成果：**

  * [代码与文档（Transformer）](https://github.com/recomoonmoon/LLM_learning_book/blob/master/Transformer/)

---

### 3. langchain基础
* 教程：
  * [大模型基础视频教程](https://www.bilibili.com/video/BV1Bo4y1A7FU/)
  
* 学习langchain的核心概念：
  
  * [Prompt Engineer](./LangChain/1_prompt.md)
  * [Parser数据解析](./LangChain/2_parser.md)
  * [Runnable(langchain)可运行对象](./LangChain/3_Runnable.md)
  * [messages信息管理](./LangChain/4_messages.md)
  * [tools工具设计与调用](./LangChain/5_tools.md)
  * [memory记忆管理](./LangChain/6_memory.md)
  * [多模态数据](/LangChain/7_multimodality.md)
  * [LCEL](/LangChain/8_LCEL.md)
  * [文档加载器](/LangChain/9_load_datas.md)
  * [文档切片器](/LangChain/10_textSplitter.md)
  * [embedding模型](/LangChain/11_embedding.md)
  * [检索器和rag](./LangChain/12_retriever.md)

* **成果：**
  * [代码与文档](https://github.com/recomoonmoon/LLM_learning_book/blob/master/LangChain/)
* **学习目标：**
  * [实现医药问答agent](https://github.com/recomoonmoon/LLM_learning_book/blob/master/LangChain/medical_qa_agent)
---
### 4. 大模型之基础 (CS336)
* 参考: [斯坦福CS336](https://online.stanford.edu/courses/cs336-language-modeling-scratch)
* 重点理解：
  * [Transformer](./CS336/Transformer/Transformer.md)
  * [BPE Tokenization](https://github.com/recomoonmoon/LLM_learning_book/blob/master/CS336/Tokenization)  
  * [Generate](./CS336/Tokenization/generate.md)
  * 工作流程
---

### 5. [大模型之训练与微调](./TrainAndFinetune/)
大模型的预训练组相关HC门槛极高而且学习很需要资源，作者只是一个database组小硕，所以我们主要进行微调学习。

* 学习路线：
  * [开源模型的本地部署](./TrainAndFinetune/download_model.py)
  * [模型的generate流程](./TrainAndFinetune/generate.py)
  * 全参数微调
  * Adapter/Prefix-Tuning
  * LoRA（Low-Rank Adaptation）
  * PEFT（Parameter-Efficient Fine-Tuning）

* 目标：在已有大模型上，快速适配特定任务。

---

### 6. 大模型之 RAG（Retrieval-Augmented Generation）

* 学习内容：

  * 向量数据库（如 FAISS, Milvus）
  * 文档检索 + LLM 推理的结合
  * 知识增强型对话与问答系统

---

### 7. 大模型之 Agent

* 学习内容：

  * ReAct 框架（Reason + Act）
  * 工具调用（Tool Use）
  * 多步推理（Chain-of-Thought）
  * 自主任务分解与执行

* 目标：让 LLM 从单纯对话扩展为 **能完成复杂任务的智能体**。

---

## 备注

本学习路线持续更新中，代码与文档将同步在 [GitHub 仓库](https://github.com/recomoonmoon/LLM_learning_book) 中。

---
 