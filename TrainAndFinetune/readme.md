好的✅ 我帮你把文档重新整理了一下，排版清晰、目录带链接、文字更简洁流畅，便于学习和查阅。

---

# 第五部分：训练与微调

我们正式进入 **第五部分：训练与微调**。

目前已有许多训练和微调工具，其中最著名的就是 **[LLaMA-Factory](https://llamafactory.readthedocs.io/zh-cn/latest/)**。它配置好文件和数据即可完成训练，非常好用。但为了学习目的，我们先从原理出发，理解如何进行训练和微调，再学习如何使用这些工具。

由于项目和就业场景主要依赖开源模型，本节重点以 **Qwen 系列模型** 为主进行讲解与实践。

---

## 📚 参考资料

* [LLaMA-Factory 官方文档](https://llamafactory.readthedocs.io/zh-cn/latest/)
* [Qwen3 快速入门文档](https://qwen.readthedocs.io/zh-cn/latest/getting_started/quickstart.html)

---

## 📑 目录

1. [环境准备](#环境准备)
2. [数据处理](0_data_schema.md)
   * [Alpaca 格式](./0_data_schema.md#Alpaca 格式)
   * [ShareGPT 格式](./0_data_schema.md#sharegpt-格式)
3. [模型的下载与推理](./1_download_and_generate.md)
4. [预训练](#预训练)
5. [微调](#微调)

---

## 环境准备

* Python ≥ 3.10
* PyTorch ≥ 2.6
* `transformers` ≥ 4.51.0

---

 
  