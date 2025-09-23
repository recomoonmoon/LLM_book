 
---
#  CS336课程导引

## 建议学习路径
本目录主要实现了BPE Tokenizer，transformer，以及lamma和gpt架构大模型generate的代码。

建议学习顺序，跟着文档走

* [1.基础知识 ](./Tokenization/main.py)
* [2.BPE Tokenizer模型及其训练](./Tokenization/Tokenizer.md)
* [3.Transformer ](./Transformer/Transformer.md)
* [4.generate](./Tokenization/generate.md)

代码并没有实现课程文档内的所有内容，因为其中部分比较基础，所以省去，有兴趣的同学可以进行补充。

PS:部分markdown插件不能正确显示数学公式，建议复制对应章节内容让gpt教，gpt可以读懂然后正确显示并进行教学。
## 📖 目录（翻译）

1. **作业概览 (Assignment Overview)**

2. **字节对编码（Byte-Pair Encoding, BPE）分词器**
   

   * [Unicode 标准](./Tokenization/main.py)
   * [Unicode 编码方式](./Tokenization/main.py)
   * [子词（Subword）分词](./Tokenization/Tokenizer.md) 
   * BPE 分词器训练 
   * BPE 分词器实验
   * BPE 分词器：编码与解码

     * 文本编码
     * 文本解码

3. **实验部分 (Experiments)**
  实验需要准备 TinyStoriesV2数据集

4. **[Transformer 语言模型架构](./Transformer/Transformer.md)**

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

 好的，我来翻译 **第四章 Transformer 语言模型架构** 部分。

---
 
---

# 第三章 实验 (Experiments)

在这一章中，你需要运行实验来验证自己实现的 **BPE 分词器** 和 **Transformer 语言模型** 是否正确工作，并且复现出类似论文或教材中的结果。

---

## 3.1 TinyStories 上的实验

1. **数据集**

   * 使用 TinyStories 数据集（Eldan & Li, 2023）。
   * 数据由儿童故事组成，规模小、容易快速实验。

2. **任务目标**

   * 先在 TinyStories 上训练 BPE 分词器。
   * 再用该分词器训练一个小规模的 Transformer LM。

3. **评估指标**

   * **压缩率 (compression ratio)**：平均每个 token 对应多少字节。
   * **困惑度 (perplexity)**：标准语言建模指标，越低越好。

4. **实验要求**

   * 记录训练时间、内存占用。
   * 尝试不同超参数（如 vocab size, d\_model）。
   * 对比不同分词器的效果。

---

## 3.2 OpenWebText 上的实验

1. **数据集**

   * 使用 OpenWebText 的一个子集（Gokaslan & Cohen, 2019）。
   * 这是 GPT-2 使用过的数据集，规模较大，更接近真实场景。

2. **任务目标**

   * 在该数据集上训练更大规模的分词器（例如 vocab=32k）。
   * 训练一个更大的 Transformer LM。

3. **实验要求**

   * 记录耗时与 GPU/CPU 占用。
   * 分析性能瓶颈（是分词、数据加载，还是模型训练）。
   * 对比不同 vocab size 的压缩率与 perplexity。

---

## 3.3 消融实验 (Ablation Studies)

1. **为什么做消融实验？**

   * 验证模型的某个设计是否必要。
   * 例如：去掉特殊 token、减少层数、换归一化方法。

2. **具体要求**

   * 至少做一个消融实验。
   * 比较实验结果，并解释差异。

---

## 3.4 结果复现与报告

* 你需要写一份简短实验报告，总结：

  * 分词器训练结果（词表大小、压缩率）。
  * 语言模型训练结果（困惑度、泛化性能）。
  * 消融实验发现。

* 报告格式：

  * 可以是 Markdown 文件。
  * 或者写在代码注释 / Jupyter Notebook 里。

---

 
## 第四章 Transformer 语言模型架构（翻译）

### Transformer LM

在本作业中，我们将实现一个标准的 **Transformer 语言模型 (LM)**。
模型由以下几个部分组成：

1. **嵌入层 (Embedding layer)**：将 token IDs 转换为向量表示。
2. **一系列 Transformer 块**：每个块包含多头自注意力和前馈网络。
3. **最终线性层与 softmax 输出**：预测下一个 token 的概率分布。

---

### Token Embeddings（词嵌入）

* 输入是整数 ID（来自 BPE 分词器）。
* 每个 ID 被映射到一个维度为 `d_model` 的向量。
* 嵌入矩阵大小为 `(vocab_size, d_model)`。
* 我们还需要加上 **位置编码 (positional embeddings)**，以便模型感知序列顺序。

---

### Pre-norm Transformer Block（预归一化 Transformer 块）

* 每个 Transformer 块由两部分组成：

  1. **多头自注意力层 (Multi-head Self Attention)**
  2. **前馈网络 (Feedforward Network, FFN)**
* 每个子层之前应用 **RMSNorm（均方根归一化）**，再进入子层。
* 使用 **残差连接**：

  $$
  x = x + \text{Sublayer}(\text{Norm}(x))
  $$

---

### 输出归一化与嵌入（Output Normalization & Embedding）

* 在最后一个 Transformer 块之后，应用一个归一化层。
* 接着是一个 **线性层 (Linear Layer)**，其权重共享输入嵌入矩阵（即 **权重绑定 weight tying**）。
* 该线性层将隐藏状态投影到词表大小的向量，用于 softmax。

---

### 备注：批处理、Einsum 与高效计算

* 在实现注意力和 FFN 时，需要关注 **矩阵乘法效率**。
* 推荐使用 `torch.einsum` 或 `torch.matmul`，确保维度清晰、可读性高。
* 需要支持批处理输入 `(batch_size, seq_len, d_model)`。

---

### 数学符号与内存布局

* 输入张量形状： `(B, T, C)`

  * `B = batch_size`
  * `T = 序列长度`
  * `C = d_model`
* 注意力机制需要计算 `Q, K, V`：

  * `Q = X W_Q`
  * `K = X W_K`
  * `V = X W_V`
* 注意力权重：

 ![img.png](img.png) 
* 在实现时，需要确保张量维度对齐，并避免不必要的复制。

---
 
---

# 第五章 基本构建模块 (Basic Building Blocks)

在实现完整的 Transformer 语言模型之前，我们需要先构建一些 **核心模块 (building blocks)**。这些模块是神经网络的最小单元，会在整个模型中被多次复用。

---

## 5.1 线性层 (Linear Layer)

1. **功能**

   * 线性层实现仿射变换：

     ![img_1.png](img_1.png)

     其中 $x$ 是输入，$W$ 是权重矩阵，$b$ 是偏置。

2. **实现要求**

   * 使用 `torch.nn.Module` 自定义实现，而不是直接用 `nn.Linear`。
   * 参数初始化：

     * 权重 $W$ 使用 **Kaiming 初始化** 或 **Xavier 初始化**。
     * 偏置 $b$ 初始化为 0。

3. **输入输出形状**

   * 输入：`(B, T, in_dim)`
   * 输出：`(B, T, out_dim)`

---

## 5.2 嵌入层 (Embedding Layer)

1. **功能**

   * 将 token ID（整数）映射到向量表示。
   * 实现类似 `nn.Embedding` 的功能。

2. **实现细节**

   * 参数矩阵形状 `(vocab_size, d_model)`。
   * 输入是 `(B, T)` 的整数张量，输出是 `(B, T, d_model)`。

3. **权重共享 (Weight Tying)**

   * 在语言模型中，嵌入矩阵会与输出层线性变换共享权重。
   * 这样可以减少参数数量，并提升泛化性能。

---

## 5.3 RMSNorm（均方根归一化）

1. **为什么不用 LayerNorm？**

   * LayerNorm 依赖于均值和方差，计算开销较大。
   * RMSNorm 只使用均方根 (root mean square)，计算更高效。

2. **定义**
   给定输入向量 $x$，RMSNorm 的输出为：

  ![img_2.png](img_2.png)

   * $g$ 是可学习的缩放参数。
   * $\epsilon$ 是数值稳定性常数。

3. **实现要求**

   * 不要用 PyTorch 自带的 `nn.LayerNorm`。
   * 自己实现 RMSNorm，并支持 `(B, T, C)` 的输入。

---

## 5.4 前馈网络 (Feedforward Network, FFN)

1. **结构**

   * FFN 通常包含两层线性变换和一个激活函数：

    ![img_3.png](img_3.png)

2. **细节**

   * 隐藏层维度一般取 $4 \times d_{model}$。
   * 激活函数使用 **GELU** 或 **ReLU**。

3. **实现要求**

   * 输入形状 `(B, T, d_model)`，输出相同。
   * 确保高效支持批处理。

---
好，我们接着翻译 **第六章：注意力机制 (Attention Mechanism)**。

---

# 第六章 注意力机制 (Attention Mechanism)

注意力机制是 Transformer 的核心。它允许模型在预测下一个 token 时“关注”输入序列中的相关位置，而不是仅依赖固定的上下文窗口。

---

## 6.1 自注意力 (Self-Attention)

1. **定义**
    
   ![img_7.png](img_7.png)

2. **注意力权重**

  ![img_6.png](img_6.png)

3. **输出**

   * 最终输出为：

    ![img_5.png](img_5.png)

---

## 6.2 缩放点积注意力 (Scaled Dot-Product Attention)

* 之所以要除以 $\sqrt{d_k}$，是为了避免在高维空间中点积过大，导致 softmax 进入梯度消失区间。
* 在实现中：

  * 输入形状 `(B, T, d_model)`。
  * 输出形状 `(B, T, d_model)`。

---

## 6.3 掩码 (Masking)

1. **因果掩码 (Causal Mask)**

   * 在语言模型中，预测位置 $t$ 不能看到未来的 token。
   * 因此，需要掩盖未来位置：

     $$
     \alpha_{ij} = 0 \quad \text{if } j > i
     $$

2. **实现方式**

   * 使用一个上三角矩阵，mask 掉未来的 logits。
   * 在 PyTorch 中，可以用 `torch.triu` 构造掩码。

---


## 6.4 多头注意力 (Multi-Head Attention)

1. **动机**

   * 单一的注意力头可能无法捕捉到所有依赖关系。
   * 多头机制允许模型在不同子空间中学习不同的注意力模式。

2. **实现**

   * 将输入投影到 $h$ 个子空间：

     $$
     Q_h, K_h, V_h
     $$
   * 在每个子空间独立计算注意力。
   * 将所有头的结果拼接，再通过线性层投影回原维度。

3. **公式**

   ![img_4.png](img_4.png)
---

## 6.5 实现注意事项

* **张量形状**：

  * 输入：`(B, T, d_model)`
  * 经过拆分头部后：`(B, h, T, d_head)`
  * 输出拼接回 `(B, T, d_model)`

* **高效计算**

  * 尽量使用 `einsum` 或 `matmul` 避免显式 for 循环。
  * 注意保存中间结果，减少重复计算。

---

# 第7章 实验 

**来源：** *cs336\_spring2025\_assignment1\_basics.pdf*。下文翻译严格基于该文档第 7 部分内容（见引用）。&#x20;

> 说明：我把第 7 章（`7 Experiments`）的全部小节按原文逻辑与顺序逐条、逐句翻译为中文并以 Markdown 排版。为便于核对与后续查阅，每个小节后面我都放了对应的文档引用标记（来自上传的 PDF）。如需我把翻译与原文并排呈现或输出为单独的 `.md` 文件，可告诉我我会直接生成下载链接。

---

## 目录

1. [总述 / 引言](#总述--引言)
2. [7.1 如何运行实验与提交项（How to Run Experiments and Deliverables）](#71-如何运行实验与提交项how-to-run-experiments-and-deliverables)
3. [7.2 TinyStories（小数据集实验）](#72-tinystories小数据集实验)

   * 超参数建议与调优
   * 示例文本
   * 可交付成果（Deliverables）
4. [7.3 消融（Ablations）与架构改动（Architecture modifications）](#73-消融ablations与架构改动architecture-modifications)

   * 消融 1：Layer Normalization（RMSNorm）
   * 消融：Pre-norm vs Post-norm
   * 消融 2：位置编码（RoPE vs NoPE）
   * 消融 3：SwiGLU vs SiLU
   * 提交要求
5. [7.4 在 OpenWebText 上运行（Running on OpenWebText）](#74-在-openwebtext-上运行running-on-openwebtext)

   * 示例与注意事项
   * 主实验说明与可交付成果
6. [7.5 自定义改进与排行榜（Your own modification + leaderboard）](#75-自定义改进与排行榜your-own-modification--leaderboard)

   * 排行榜规则
   * 参考资源与提示
   * 可交付成果
7. [实用小贴士与低资源选项（摘录）](#实用小贴士与低资源选项摘录)

---

## 总述 / 引言

现在是把前面各个模块整合起来、训练（小型）语言模型并运行一系列实验的时候了。实验的目标是以动手修改和运行为主，通过在可控的小规模设置上反复试验，来理解 Transformer 各组成部分对训练稳定性与最终效果的影响。为此课程建议先在小模型（约 17M 参数）与简单数据集（TinyStories）上快速实验，再迁移到更大、噪声更多的 OpenWebText 数据集上做对比与扩展。&#x20;

---

## 7.1 如何运行实验与提交项（How to Run Experiments and Deliverables）&#x20;

**要点（翻译并归纳）：**

* 最好的学习方式是“改动并亲自运行”——通过亲自修改 Transformer 并训练来理解各个组件的作用。
* 实验应做到：快速可复现（快速在小规模上跑实验）、系统化（按计划做消融与超参扫描）、并记录详细日志（实验尝试、学习曲线等）。
* 为便于提交损失曲线，务必周期性地计算并记录**验证损失（validation loss）**，同时记录梯度步数和实时时钟（wallclock time）。推荐使用像 Weights & Biases 这样的实验追踪工具以便记录。

**任务（Problem (experiment\_log)，3 分）：**

* 为训练与评估代码构建实验追踪基础设施（能够按梯度步数和真实时间记录学习曲线）。
* **可交付物（Deliverable）**：把用于实验的日志基础设施代码提交，并提交一份“实验日志”（记录你尝试过的所有设置与结果）。&#x20;

---

## 7.2 TinyStories（小型数据集实验）

### 说明

* 本节先使用 TinyStories（Eldan & Li, 2023）这个非常小且易训练的数据集，让模型能很快收敛并可以看到一些有趣行为。数据集获取方式见文档第 1 节。下面给出一个 TinyStories 的样例，以便对数据有直观感受（文档中给了示例段落）。&#x20;

**示例（文档原示例已在源文件）：**

> （示例文本略长——文档中给出了一段完整的 TinyStories 样本，供生成质量对比使用。）&#x20;

### 超参数建议（起始值）

文档给出了一组用来启动训练的基本超参（**你应当以此为起点并做搜索**）：

* `vocab_size`: **10000**。通常词表会从几万到几十万，建议试验不同词表尺寸观察压缩率与模型行为变化。
* `context_length`（上下文长度 / sequence length）: **256**。TinyStories 不需要太长的上下文，但后续 OpenWebText 可能要更长。
* `d_model`: **512**（隐层维度）。比常见的 768 略小，以便训练更快。
* `d_ff`: **1344**（Feed-Forward 隐层维度；约等于 $\tfrac{8}{3} d\_model$ ，同时取能被 64 整除以更好利用 GPU tensor core）。
* RoPE（旋转位置编码）参数 `Θ`: **10000**。
* 层数 / heads: **4 层, 16 heads**（对应约 17M 非嵌入参数，是一个较小的 Transformer 配置）。
* **总处理 token 数（total tokens processed）**：**327,680,000**（你的 batch\_size × total step count × context\_length 应大致等于这个值）。

**需你调优的超参（建议做搜索）：** 学习率（learning rate）、warmup 步数、AdamW 的 β1/β2/ϵ、weight decay 等。文中建议参考 Kingma & Ba（Adam 原文）来选择典型值。

### 运行时间提示

* 如果实现正确且高效，上述超参在单块 H100 GPU 上大约需要 **30–40 分钟**。如果你的 runtime 大幅超过该范围，请检查数据加载、checkpoint 或验证逻辑是否成为瓶颈，并确保实现正确地使用批处理。&#x20;

### 可交付成果（涉及 TinyStories 的任务）

* **训练脚本与模型**：按要求训练 BPE tokenizer、对数据分词、运行训练循环并周期性保存验证损失。
* **生成文本（Problem: generate）**：用训练好的模型生成文本，要求至少输出 256 个 token（或直到 `<|endoftext|>`），并对生成流畅性给出简短评述与至少两个影响质量的因素。&#x20;

---

## 7.3 消融（Ablations）与架构改动（Architecture modification）

> 本小节目的：通过有针对性的“消融实验（ablation）”来理解 Transformer 各部分对训练稳定性与性能的影响。每个消融是一个独立的小实验，要求你做对照（baseline vs 修改后的模型）并上交学习曲线与结论。

### 消融 1：Layer Normalization（移除 RMSNorm）

**背景：** Layer normalization（此处使用 RMSNorm 的变体）常被认为对于 Transformer 的训练稳定性至关重要。
**任务（layer\_norm\_ablation）**：移除 Transformer block 中所有的 RMSNorm 并用之前的最优学习率训练。
**要回答的问题：** 在之前的最优学习率下训练会发生什么？是否需要把学习率降低以恢复稳定性？
**可交付成果：**

* 移除 RMSNorm 后的学习曲线；
* 在调整后最佳学习率下的学习曲线；
* 简短讨论 RMSNorm 对训练的影响。&#x20;

### 消融：Pre-norm vs Post-norm（层归一化位置）

**背景：** Pre-norm（课程默认）定义为：

```
z = x + MultiHeadedSelfAttention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

原始 Transformer 使用的是 post-norm：

```
z = RMSNorm(x + MultiHeadedSelfAttention(x))
y = RMSNorm(z + FFN(z))
```

**任务（pre\_norm\_ablation）**：把实现从 pre-norm 改为 post-norm，训练并比较两种设计的差异。
**可交付成果：** post-norm 与 pre-norm 的学习曲线对比与简短分析。&#x20;

### 消融 2：位置编码（RoPE vs NoPE）

**背景：** 探究位置编码对模型性能的影响：把带 RoPE（旋转位置编码）的基线模型与\*\*没有任何位置编码（NoPE）\*\*的模型做对比。文献表明在 decoder-only（因果掩码）设置下，模型理论上可从纯数据中推断相对/绝对位置信息，但实证表现值得检验。
**任务（no\_pos\_emb）**：移除位置信息（NoPE）并训练对比。
**可交付成果：** RoPE vs NoPE 的学习曲线对比与讨论。&#x20;

### 消融 3：SwiGLU vs SiLU（FFN 中 Gating 重要性）

**背景：** 根据 Shazeer（2020）关于门控机制的讨论，测试带有门控的 SwiGLU 与普通的 SiLU 激活的 FFN 的差异。文档给出参数匹配的建议：

* SwiGLU 的内层宽度采用 $d\_{ff} \approx \frac{8}{3} d\_{model}$（并保证 `d_ff % 64 == 0`）；
* 为了给 SiLU 网络匹配参数量，建议把 SiLU FFN 设为 `d_ff = 4 * d_model`（因为 SiLU FFN 只有两个权重矩阵而 SwiGLU 有三个）。
  **任务（swiglu\_ablation）**：实现并比较 SwiGLU 和 SiLU 的表现（参数量近似匹配）。
  **可交付成果：** 两者的学习曲线，及简短结论。&#x20;

### 其它消融或改动建议

* 还有建议的改动包括：权重初始化、embedding / LM head 权重绑定（weight tying）、不同归一化方式等（这些可以作为你自己的实验项在排行榜/自定义改动中提交）。文中也建议先在 TinyStories 上测试改动再转到 OpenWebText。&#x20;

---

## 7.4 在 OpenWebText 上运行（Running on OpenWebText）&#x20;

### 说明

* OpenWebText（OWT）是一个更接近真实 web 文本的预训练语料（由 Web 抓取构成），文档提供了一个小型样例供查看（片段更复杂、现实）。在转到 OWT 做实验时，**可能需要重新调节学习率 / batch size 等超参**。示例片段用于观察训练数据的复杂性与风格。&#x20;

**任务（main\_experiment）：在 OWT 上做实验（2 分，约 3 H100 小时）**

* 用与 TinyStories 相同的模型架构与训练迭代次数在 OpenWebText 上训练模型，观察模型在更真实/复杂语料上的表现。

**可交付成果：**

1. 在 OpenWebText 上的学习曲线（validation loss 相对于训练步数 & 实时时钟）；讨论与 TinyStories 的损失差异以及这些差异如何被解释（比如语料难度、噪声、domain 差异等）。
2. 用训练好的 OWT 模型生成文本（与 TinyStories 的生成格式一致），输出不少于 256 token 的文本样本，并评价生成的流畅度及为什么即使与 TinyStories 使用同样模型和计算预算，生成质量可能更差。&#x20;

---

## 7.5 自定义改进与排行榜（Your own modification + leaderboard）

### 目标

* 在此阶段，你尝试对 Transformer 架构做改进（或改变超参 / 训练技巧），在课程排行榜上与其他同学比较结果。该部分鼓励创造性尝试（但有几条必要规则）。

### 排行榜规则（关键限制）

* **运行时间限制（Runtime）**：你的提交最多只能在 **H100 上运行 1.5 小时**。提交脚本可通过 `--time=01:30:00` 在 slurm 中强制限制。
* **数据限制（Data）**：你只能使用我们提供的 **OpenWebText** 训练数据集。
* 其它方面自由发挥（可尝试模型改进、预处理、优化器变体等）。&#x20;

### 参考资源（给想法的方向）

* 参考开源的 SOTA LLM（如 Llama 3、Qwen 2.5）的设计与技巧；
* NanoGPT 的 speedrun / modded 分支（社区在小规模训练上已实现很多“快跑”改动），例如权重绑定（weight tying），但如果使用 weight tying 可能要调整 embedding/LM head 初始化标准差等。建议在小数据集或小子集上先测试再跑完整提交。&#x20;

### 提交要求（Problem: leaderboard）

* 在 1.5 小时 H100 预算内训练你的模型以最小化验证 loss（目标是优于基线）。
* **可交付成果**：记录的最终验证 loss、学习曲线（x 轴为实时时钟并且小于 1.5 小时）、以及简要描述你所做的改动与实验设置。教师期望排行榜提交能跑出低于默认朴素基线（例如 loss 5.0）的成绩。提交到课程排行榜仓库（文档给出了 GitHub 链接）。&#x20;

---

## 实用小贴士与低资源选项（摘录）

* **低资源/缩放建议（Low-Resource/Downscaling Tips）**：如果 GPU 资源有限，可在 CPU 或 Apple Silicon（M 系列）上做较短的训练（例如把 total tokens 降到 40M），并把目标验证 loss 放宽（例如从 1.45 放宽到 1.80 或 2.0），这个配置能在 CPU/MPS 上得到合理的训练时间与结果对比。文中给出了在 M3 Max 的实验数据作为参考（例如某配置在 CPU 上耗时 \~1 小时 22 分）。

* **调试建议**：

  * 先尝试在单个 minibatch 上过拟合（overfit）以检查模型实现是否正确；
  * 在模型各组件处设置断点查看 tensor 形状；
  * 监控激活、参数与梯度范数以避免爆炸或消失。&#x20;

* **学习率、batch size 实验**：课程鼓励做学习率的 sweep（寻找“边界的最佳学习率”），并做不同 batch size 的比较实验（从 1 到 GPU 内存上限），记录学习曲线并讨论发现。

---

## 小结（如何按文档要求开展作业 / 实验流程建议）

1. **先完成 TinyStories 的训练流水线**：实现 tokenizer -> 数据序列化 -> 模型 -> 训练循环 -> checkpoint 与验证评估。用文档给的超参作为起点。&#x20;
2. **实现实验日志跟踪（experiment\_log）**：周期性上报训练/验证损失与 wallclock，保存学习曲线。&#x20;
3. **按文档的多个“问题（Problem）”逐一实现并提交可交付物**（例如 decoding、learning rate sweep、batch size 比较、各种消融、TinyStories 与 OpenWebText 的主实验）。
4. **最终做排行榜提交（选做）**：在 1.5 小时 H100 限制内实现你认为有效的改进并提交学习曲线与最终 loss。&#x20;

---

 
---
 
---

# 第八部分：训练 Transformer 语言模型

我们已经完成了数据预处理（通过分词器）和模型（Transformer）的实现。接下来需要搭建支持训练的所有代码。这包括以下几个部分：

* **损失函数（Loss）**：我们需要定义损失函数（交叉熵）。
* **优化器（Optimizer）**：我们需要定义优化器以最小化损失（AdamW）。
* **训练循环（Training Loop）**：我们需要所有辅助基础设施，用于加载数据、保存检查点以及管理训练。

---

## 8.1 交叉熵损失（Cross-entropy Loss）

回忆一下，Transformer 语言模型为每个长度为 $m+1$ 的序列 $x$ 定义了一个分布

$$
p_\theta(x_{i+1} \mid x_{1:i}), \quad i = 1, \dots, m
$$

给定一个包含长度为 $m$ 序列的数据集 $D$，我们定义标准交叉熵（即负对数似然，negative log-likelihood）损失函数：

![img\_16.png](img_16.png)

> 注意：一次 Transformer 的前向传播会同时输出所有 $i = 1, \dots, m$ 的 $p_\theta(x_{i+1} \mid x_{1:i})$。

具体来说，Transformer 会在每个位置 $i$ 输出一个向量 logits

$$
o_i \in \mathbb{R}^{|\text{vocab}|}
$$

从而得到概率：

![img\_15.png](img_15.png)

交叉熵损失通常是针对 logits 向量 $o_i$ 与目标 token $x_{i+1}$ 定义的。

**实现提示**：实现交叉熵损失时需要注意数值稳定性（和 softmax 一样）。

---

### 问题 (cross\_entropy)

实现交叉熵损失函数。要求：

* 输入预测的 logits $o_i$ 与目标 $x_{i+1}$。
* 计算交叉熵损失

  $$
  \ell_i = - \log \text{softmax}(o_i)[x_{i+1}]
  $$
* 要求处理以下细节：

  * 减去最大值以保证数值稳定性。
  * 尽可能在实现时抵消 `log` 与 `exp`。
  * 处理 batch 维度，并返回 batch 的平均损失（batch 维度始终在 vocab 维度之前）。

**提交要求**：实现 `[adapters.run_cross_entropy]`，并运行 `uv run pytest -k test_cross_entropy` 测试。

---

### 困惑度（Perplexity）

交叉熵足以用于训练，但在评估模型时，我们通常还需要报告 **困惑度（perplexity）**。
对于一个长度为 $m$ 的序列，假设我们得到了交叉熵损失 $\ell_1, \dots, \ell_m$，则：

![img\_14.png](img_14.png)

---

## 8.2 SGD 优化器（The SGD Optimizer）

有了损失函数，我们开始探索优化器。最简单的基于梯度的优化器是 **随机梯度下降（Stochastic Gradient Descent, SGD）**。

从随机初始化参数 $\theta_0$ 开始，对于每一步 $t = 0, \dots, T-1$，我们执行以下更新：

![img\_13.png](img_13.png)

其中 $B_t$ 是从数据集 $D$ 中采样的一个 batch，学习率 $\alpha_t$ 和 batch 大小 $|B_t|$ 是超参数。

---

## 8.2.1 在 PyTorch 中实现 SGD

要在 PyTorch 中实现优化器，我们需要继承 `torch.optim.Optimizer` 类。子类必须实现两个方法：

* `__init__(self, params, …)`

  * 初始化优化器，`params` 是需要优化的参数集合。
  * 确保将 `params` 传递给基类构造函数，以便其存储这些参数供 `step` 使用。
  * 可以接收额外参数（如学习率）。

* `step(self)`

  * 执行一次参数更新。
  * 在训练循环中，会在反向传播（backward）后调用。
  * 遍历每个参数 `p`，利用 `p.grad` 更新 `p.data`。

接下来我们实现一个带 **学习率衰减** 的 SGD：

![img\_12.png](img_12.png)

---

## 8.3 AdamW

虽然 SGD 是最简单的优化器，但在现代深度学习中，我们通常使用 **Adam** 或其变体。

Adam 优化器结合了两个思想：

![img\_11.png](img_11.png)

其中：

* $\alpha$：学习率
* $\epsilon$：数值稳定性项
* $\beta_1, \beta_2$：衰减参数（常用 0.9 和 0.999）

此外，**AdamW**（“Adam with Weight Decay”）加入了权重衰减（即 L2 正则化），比原始 Adam 更适合训练 Transformer。

---

## 8.4 学习率调度（Learning Rate Scheduling）

学习率在训练中起着关键作用。

* **预热（Warm-up）**：训练初期学习率逐渐增加，避免梯度爆炸。
* **衰减（Decay）**：训练后期逐渐降低学习率，使模型收敛到更优点。

一种常见调度策略（Transformer 论文中提出）：

![img\_10.png](img_10.png)

其中：

* $d_{\text{model}}$：隐藏层维度
* $w$：预热步数（例如 4000）
* $t$：训练步数

这种策略在训练大模型时非常有效。

---

## 8.5 梯度裁剪（Gradient Clipping）

训练深度模型时，有时梯度可能会变得非常大，导致参数更新过于激烈甚至发散。

![img\_9.png](img_9.png)

其中 $\tau$ 是阈值（例如 1.0）。

* 如果梯度过大，就按比例缩小。
* 如果梯度在范围内，则保持不变。

PyTorch 提供了 `torch.nn.utils.clip_grad_norm_` 来实现这一功能。

---
 

---

## 8.6 训练循环（Training Loop）

结合前面所有组件，我们可以搭建完整的训练循环：

1. **数据加载**

   * 使用 `torch.utils.data.DataLoader` 批量加载数据。
   * 将数据分为训练集和验证集。

2. **前向传播**

   * 输入 token IDs，Transformer 输出 logits。

3. **计算损失**

   * 使用交叉熵函数。

4. **反向传播**

   * `loss.backward()` 计算梯度。

5. **梯度裁剪**

   * 限制梯度范数。

6. **优化器更新**

   * 调用 `optimizer.step()`。

7. **学习率调度**

   * 更新学习率。

8. **清理梯度**

   * `optimizer.zero_grad()`。

---

### 检查点保存（Checkpointing）

在长时间训练中，我们需要定期保存模型和优化器的状态，以便：

* 恢复训练（避免中断）。
* 保存最优模型（基于验证集性能）。

PyTorch 提供了 `torch.save` 和 `torch.load`：

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, "checkpoint.pt")

# 加载
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

---
 
 

---

## 8.7 数据加载器（Data Loader）

为了高效地训练，我们需要把原始文本数据组织成小批量（mini-batch），并将其输入模型。

在 PyTorch 中，可以使用 `torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader`：

1. **Dataset**：定义如何从磁盘读取数据。

   * 例如，每次返回一个文本序列（输入 IDs + 标签）。
   * 你可以编写一个自定义类继承 `Dataset`。

2. **DataLoader**：将 `Dataset` 包装成批量数据，并支持 **打乱（shuffle）** 和 **多线程加载**。

一个简化示例：

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.tensor(x), torch.tensor(y)

# 构造 DataLoader
train_loader = DataLoader(TextDataset(train_ids, block_size=128), batch_size=64, shuffle=True)
```

这样，训练时每个 batch 会提供 `(x, y)` 对，供 Transformer 预测下一个 token。

---

## 8.8 检查点保存（Checkpointing）

训练语言模型可能需要数小时甚至数天，因此必须保存中间状态。

### 为什么要保存检查点？

* **防止意外中断**：如果训练被打断，可以从保存点恢复。
* **记录最优模型**：在验证集上表现最佳的模型需要保存下来。

### PyTorch 实现方式

```python
# 保存
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epoch": epoch,
    "loss": loss,
}, "checkpoint.pt")

# 加载
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
```

注意：恢复训练时需要确保模型和优化器初始化一致。

---

## 8.9 完整训练循环（Full Training Loop）

综合 **前向传播 → 计算损失 → 反向传播 → 更新参数 → 学习率调度 → 保存检查点**，得到完整流程：

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        # 前向传播
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 学习率调度
        scheduler.step()

    # 验证
    model.eval()
    val_loss = evaluate(model, val_loader)

    # 保存检查点
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }, f"checkpoint_{epoch}.pt")
```

训练过程中通常会记录 **loss 曲线** 以监控收敛情况。

---

## 8.10 小结

在本节中，我们完成了 **Transformer 语言模型的训练框架**：

* 使用 **交叉熵损失** 作为目标函数。
* 使用 **AdamW 优化器**，并结合 **学习率调度** 和 **梯度裁剪**。
* 通过 **DataLoader** 提供批量数据。
* 利用 **检查点保存**，保证训练过程稳定和可恢复。
* 设计了一个 **完整的训练循环**，支持大规模训练。

至此，我们已经具备了一个完整的 Transformer LM 训练系统，可以在合适的数据集上开始实验。

---

 

 

---
 
 

## 9.文本生成

我帮你逐段拆解这段 `decode` 代码，解释它在做什么（相当于一个简化版的 GPT 文本生成函数）：

---

### 1. 装饰器

```python
@torch.no_grad()
```

* 作用：关闭梯度计算，节省显存、加快推理速度。
* 推理阶段不需要反向传播，所以加上它。

---

### 2. 函数定义

```python
def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> str:
```

参数说明：

* `model`: Transformer 语言模型。
* `tokenizer`: 编码/解码工具，能把字符串转为 token IDs，再解码回文本。
* `prompt`: 输入的上下文。
* `max_new_tokens`: 最多生成多少新 token。
* `temperature`: 控制随机性，>1 更随机，<1 更确定。
* `top_p`: nucleus sampling 的阈值，保留累计概率 ≤ p 的 token。
* `device`: GPU/CPU。

---

### 3. 初始化

```python
model.eval()
model.to(device)

input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

eos_id = tokenizer.encode("<|endoftext|>")[0]
```

* `model.eval()`：切换到推理模式（关闭 dropout 等）。
* `input_ids`：把 prompt 编成 token id，并放入 GPU/CPU。
* `eos_id`：取 `<|endoftext|>` 的 token id，用来判断是否结束生成。

---

### 4. 逐步生成 token

```python
for _ in range(max_new_tokens):
    logits = model(input_ids)        # (1, seq_len, vocab_size)
    logits = logits[:, -1, :]        # 取最后一个位置的预测分布
```

* `model(input_ids)`：前向计算，得到每个位置的词预测 logits。
* 只取最后一个位置的 logits，因为我们只关心“接下来可能的下一个词”。

---

### 5. 应用温度系数

```python
logits = logits / max(temperature, 1e-8)
probs = F.softmax(logits, dim=-1)
```

* 温度缩放：

  * 如果 `temperature < 1` → 分布更尖锐，更确定。
  * 如果 `temperature > 1` → 分布更平滑，更随机。
* `softmax` 把 logits 转为概率分布。

---

### 6. Top-p (核采样)

```python
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

cutoff = cumulative_probs > top_p
cutoff[..., 1:] = cutoff[..., :-1].clone()
cutoff[..., 0] = False
```

* `torch.sort`：按概率从大到小排序。
* `cumulative_probs`：累计概率。
* `cutoff`: 标记累计概率超过 `top_p` 的部分。

  * 例如 `top_p=0.9`，则只保留累计概率 ≤0.9 的 token。
  * 这样避免长尾的低概率 token。
* `cutoff[..., 0] = False`：保证至少保留最高概率的那个词。

---

### 7. 归一化 & 采样

```python
sorted_probs[cutoff] = 0
sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

next_token = torch.multinomial(sorted_probs, num_samples=1)
next_token = sorted_indices.gather(-1, next_token)
```

* 把超出 top\_p 的 token 概率置零。
* 重新归一化。
* `torch.multinomial`：按分布采样一个 token id。
* `sorted_indices.gather`：把采样结果映射回原始 token id。

---

### 8. 拼接 & 停止条件

```python
input_ids = torch.cat([input_ids, next_token], dim=-1)

if next_token.item() == eos_id:
    break
```

* 把新 token 拼到已有序列后面。
* 如果采样到 `<|endoftext|>`，则提前结束。

---

### 9. 解码返回

```python
return tokenizer.decode(input_ids[0].tolist())
```

* 把最终 token id 序列解码成字符串。
* 输出包含 prompt + 生成的新内容。

---

✅ **总结一下**
这段 `decode` 就是一个 **逐 token 生成文本的推理循环**，流程是：

1. 输入 prompt → 编码成 token ids
2. 模型预测下一个词的概率分布
3. 应用温度缩放 + Top-p 过滤
4. 从分布中采样一个 token
5. 拼接到序列，继续预测
6. 如果遇到 `<|endoftext|>` 或生成到上限，就停止
7. 解码成字符串输出

它实现了 GPT 类模型常见的 **温度采样 + nucleus sampling** 的文本生成。

---

 
 