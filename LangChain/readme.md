### Langchain基础
* 教程：
  * [大模型基础视频教程](https://www.bilibili.com/video/BV1Bo4y1A7FU/)
  * [langchain官方文档](https://python.langchain.com/docs/concepts/)
  
* 环境依赖
  * openai 调用API
  * dotenv 存储
  * langchain Agent开发基础
  
* 主要内容
  * [Prompt Engineer](./1_prompt.md)
  * [Parser数据解析](./2_parser.md)
  * [Runnable(langchain)可运行对象](./3_Runnable.md)
  * [messages信息管理](./4_messages.md)
  * [tools工具设计与调用](./5_tools.md)
  * [memory记忆管理](./6_memory.md)
  * [多模态数据](7_multimodality.md)
  * [LCEL](8_LCEL.md)
  * [文档加载器](9_load_datas.md)
  * [文档切片器](10_textSplitter.md)
  * [embedding模型](11_embedding.md)
  * [检索器与rag](12_retriever.md)
  * [事件触发器](13_callBack.md)
* 章节作业：
  * 利用提示词模板 + RAG + Parser + 消息管理 + tool 实现一个Agent。
  * 模板使用的是医药问答agent
  * 各个技术不一定要拘于形式地使用，比如rag可以key为关键词，value为索引，然后数据存储在关系数据库等地方。这样可以通过多属性索引快速查找相关数据又减少存储。
  
---
