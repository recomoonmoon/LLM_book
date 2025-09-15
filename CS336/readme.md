 
---
#  CS336课程文件翻译

## 📖 目录（翻译）

1. **作业概览 (Assignment Overview)**

2. **字节对编码（Byte-Pair Encoding, BPE）分词器**

   * Unicode 标准
   * Unicode 编码方式
   * 子词（Subword）分词
   * BPE 分词器训练
   * BPE 分词器实验
   * BPE 分词器：编码与解码

     * 文本编码
     * 文本解码

3. **实验部分 (Experiments)**

4. **Transformer 语言模型架构**

   * Transformer LM
   * 词嵌入 (Token Embeddings)
   * 预归一化 Transformer 块 (Pre-norm Transformer Block)（做了，跳过）
   * 输出归一化与嵌入（做了，跳过）
   * 备注：批处理、Einsum 与高效计算
   * 数学符号与内存布局

5. **基本构建模块：线性与嵌入层**

   * 参数初始化
   * 线性层 (Linear Module)
   * 嵌入层 (Embedding Module)

6. **预归一化 Transformer 块**

   * 均方根层归一化 (RMSNorm)
   * 前馈网络 (FFN)
   * 相对位置嵌入
   * 缩放点积注意力
   * 因果多头自注意力

7. **完整 Transformer LM**

8. **训练 Transformer LM**

   * 交叉熵损失函数
   * SGD 优化器
   * 在 PyTorch 中实现 SGD
   * AdamW
   * 学习率调度
   * 梯度裁剪
   * 训练循环
   * 数据加载器
   * 检查点保存
   * 训练循环

9. **文本生成 (Generating text)**

10. **实验与交付物 (Experiments & Deliverables)**

* TinyStories 数据集
* Ablation 实验与架构修改
* OpenWebText 实验
* 自定义修改 + 排行榜

---

## 📖 第一章：作业概览（翻译）

### 作业目标

在本次作业中，你将从零开始构建训练标准 **Transformer 语言模型 (LM)** 所需的全部组件，并完成训练。

### 你需要实现的部分

1. **字节对编码 (BPE) 分词器** (§2)
2. **Transformer 语言模型** (§3)
3. **交叉熵损失函数 与 AdamW 优化器** (§4)
4. **训练循环**（支持模型与优化器状态的保存与加载） (§5)

### 你需要运行的任务

1. 在 **TinyStories 数据集** 上训练一个 BPE 分词器。
2. 使用训练好的分词器将数据集转换为整数 ID 序列。
3. 在 TinyStories 上训练一个 Transformer 语言模型。
4. 使用训练好的 Transformer LM 生成样例文本，并评估困惑度 (perplexity)。
5. 在 **OpenWebText** 上训练模型，并将困惑度提交到排行榜。

### 使用规范

* 要求你 **从零实现** 这些组件。
* 你 **不可以** 使用 `torch.nn`, `torch.nn.functional`, `torch.optim` 中的任何现成定义，除了：

  * `torch.nn.Parameter`
  * 容器类（如 `Module`, `ModuleList`, `Sequential` 等）
  * `torch.optim.Optimizer` 基类
* 你可以使用 PyTorch 的其他定义。
* 如果不确定某个函数/类能否使用，可以在 Slack 上提问。原则是：不要破坏“从零实现”的精神。

### 关于 AI 工具

* 允许使用 ChatGPT 等 LLM 提问 **低层编程问题** 或 **高层概念问题**。
* 禁止直接让 AI **代写作业**。
* 建议关闭 IDE 中的 AI 自动补全（如 Copilot），以便更深入理解作业内容。

### 代码说明

* 作业代码与说明文档托管在 GitHub：
  👉 [github.com/stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
* 仓库结构：

  1. `cs336_basics/*`：你要写的代码（为空，可以自由实现）。
  2. `adapters.py`：作业框架要求的功能接口，需调用你写的代码（不能在此写逻辑）。
  3. `test_*.py`：测试文件，调用 `adapters.py`，不要修改。

### 提交要求

* 上传到 Gradescope：

  * **writeup.pdf**：书面回答问题（需排版）。
  * **code.zip**：你写的所有代码。
* 提交排行榜：

  * 在 [assignment1-basics-leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard) 提交 PR。

### 数据集

* 使用两个已处理好的数据集：

  * **TinyStories** (Eldan & Li, 2023)
  * **OpenWebText** (Gokaslan et al., 2019)
* 数据集是纯文本文件，可在 `/data` 获取（课堂环境），或在 GitHub README 提供的命令下载（本地环境）。

---

  
---

# 第二章：字节对编码（BPE）分词器

在本作业的第一部分，你需要实现并训练一个 **字节级别（byte-level）的 BPE 分词器** \[Sennrich et al., 2016; Wang et al., 2019]。
核心思路：

* 将任意 Unicode 字符串表示为 **字节序列**；
* 在这些字节序列上训练 BPE 分词器；
* 最终用该分词器将字符串编码成整数序列（token IDs），供语言模型使用。

---

## 2.1 Unicode 标准

Unicode 是一个文本编码标准，用于把字符映射为整数编码点（code points）。
截至 Unicode 16.0（2024 年 9 月发布），该标准共定义了 **154,998 个字符**，覆盖 **168 种文字系统**。

* 例如：

  * 字符 “s” 的编码点是 **115**，写作 `U+0073`（`0073` 是十六进制）。
  * 汉字 “牛” 的编码点是 **29275**。

在 Python 里：

* `ord()`：字符 → 整数编码点。
* `chr()`：整数编码点 → 对应字符。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

📌 **作业题 (unicode1, 1分)**
(a) `chr(0)` 返回什么字符？
(b) 它的 `__repr__()` 和打印结果有什么区别？
(c) 当该字符出现在字符串中会发生什么？

---

## 2.2 Unicode 编码

Unicode 标准本身只定义 **字符 ↔ 编码点** 的映射，但直接用编码点训练分词器不切实际：

* 词表太大（约 15 万项）；
* 很稀疏（很多字符极少出现）。

因此，我们使用 **Unicode 编码方式**，即将字符转换为 **字节序列**。
常见编码方式：

* UTF-8（互联网主流，占网页的 98% 以上）；
* UTF-16；
* UTF-32。

在 Python 中：

* `encode("utf-8")`：字符串 → 字节串
* `list(b)`：查看字节值（0\~255 的整数）
* `decode("utf-8")`：字节串 → 字符串

示例：

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> utf8_encoded.decode("utf-8")
'hello! こんにちは!'
```

关键点：

* 字节表大小固定为 **256**（0\~255）。
* 因此，任何输入文本都能被唯一表示。
* 这避免了 OOV（out-of-vocabulary）问题。

📌 **作业题 (unicode2, 3分)**
(a) 为什么我们更倾向于用 **UTF-8 字节** 训练分词器，而不是 UTF-16 或 UTF-32？
(b) 下列函数为什么错误？给一个例子。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

(c) 给出一个不能解码为 Unicode 字符的 **2 字节序列**。

---

## 2.3 子词分词（Subword Tokenization）

* **问题**：

  * 字节级分词避免了 OOV，但序列太长。
  * 一个 10 词句子，在词级模型中可能是 10 tokens，在字节级可能要 50+ tokens。
  * 更长序列 → 更慢训练 + 长依赖问题更严重。

* **子词分词器 = 折中方案**

  * 词表比字节大，但能更好压缩序列长度。
  * 例如：若 “the” 高频出现，就可以作为单个 token 存储，而不是 \[t, h, e] 三个 token。

* **BPE（Byte-Pair Encoding）**

  * 最初是一种压缩算法（Gage, 1994）。
  * 原理：迭代合并出现频率最高的字节对 → 新 token。
  * 高频词逐渐变成单个子词单元。

👉 最终得到的就是 **BPE 分词器 (BPE Tokenizer)**。

---

## 2.4 BPE 分词器训练

主要步骤：

1. **词表初始化**

   * 初始词表：所有 256 个字节。
   * 一一映射：字节 → 整数 ID。

2. **预分词（Pre-tokenization）**

   * 直接全语料统计字节对频率太慢。
   * 解决方案：先进行 **粗分词**（如用正则分词），加速统计。
   * GPT-2 使用的正则模式：

     ```regex
     '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'
     ```

3. **计算 BPE 合并**

   * 找出出现次数最多的字节对 `(A, B)`，合并为新 token `AB`。
   * 重复此过程，直到词表达到设定大小。
   * 为保证确定性：若多组频率相同，选择 **字典序较大** 的 pair。

4. **特殊符号（Special Tokens）**

   * 如 `<|endoftext|>`，应当永远保留为单一 token，不能被拆分。

---

## 2.5 实验：训练 BPE 分词器

* 任务：在 **TinyStories 数据集** 上训练一个最大词表 10K 的 BPE 分词器。
* 任务：在 **OpenWebText 数据集** 上训练一个最大词表 32K 的分词器。
* 交付物：

  * 训练耗时、内存占用。
  * 词表中最长的 token（是否合理）。
  * 性能瓶颈分析。
* 资源需求：

  * TinyStories：≤30分钟，无需GPU，≤30GB内存
  * OpenWebText：≤12小时，无需GPU，≤100GB内存

---

## 2.6 BPE 分词器：编码与解码

### 编码流程

1. **预分词**：文本 → 预分词单元 → UTF-8 字节序列。
2. **应用合并规则**：依次执行 BPE 合并 → token IDs。

### 解码流程

* 将 token IDs 映射回字节串 → 拼接 → 解码为字符串。
* ⚠️ 如果 ID 序列无法解码（非法字节序列），用 **U+FFFD 替换符号**。

---

## 2.7 分词器实验

1. 用 **TinyStories / OpenWebText 分词器** 各自编码 10 个样本文档，计算压缩率（bytes/token）。
2. 用 TinyStories 分词器去编码 OpenWebText，比较压缩率。
3. 估算分词器吞吐量（bytes/s），推算处理 **The Pile (825GB)** 所需时间。

---

 