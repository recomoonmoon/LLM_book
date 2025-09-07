---

# 学习路线笔记：从 PyTorch 到大语言模型

这是一个有一定 AI 基础的程序员学习 **LLM（Large Language Models）** 的学习记录与路线图。

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

### 3. 大模型基础
* 教程：
  * [大模型基础视频教程](https://www.bilibili.com/video/BV1Bo4y1A7FU/)
  
* 学习大模型的核心概念：
  
  * 预训练（Pre-training）
  * Tokenizer 与词表
  * 模型规模与算力需求
  * 训练数据的构建

* **成果：**
  * [代码与文档](https://github.com/recomoonmoon/LLM_learning_book/blob/master/LLM_base/)

---
### 4. 大模型之训练

* 重点理解：

  * 数据并行（DP）、模型并行（MP）、流水线并行（PP）
  * ZeRO 优化
  * 混合精度训练（FP16/BF16）
  * 高效训练框架（如 DeepSpeed、Megatron-LM）

---

### 5. 大模型之微调

* 学习几种常见的微调方法：

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
 