 
---

# 📘 Transformer Language Model (TransformerLM) 实现笔记

本文实现了一个 **简化版 Transformer 语言模型**，包括以下核心组件：

1. **Linear**: 自定义线性层（无 bias）。
2. **Embedding**: 词嵌入层，将 token id 转换为向量。
3. **RMSNorm**: Root Mean Square LayerNorm，代替传统 LayerNorm。
4. **SwiGLU**: 前馈网络的激活函数改进版本。
5. **RoPE (Rotary Positional Embedding)**: 旋转位置编码。
6. **Multi-Head Self-Attention**: 多头自注意力机制。
7. **TransformerBlock**: 基础 Transformer 块（注意力 + FFN）。
8. **TransformerLM**: 完整的语言模型。

---

## 🔹 Linear

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, ...):
        # 线性层（无偏置）
        # y = x @ W^T
```

* 使用 `einsum` 实现矩阵乘法。
* 权重采用截断正态分布初始化，保证稳定。

---

## 🔹 Embedding

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, ...):
        # 将 token_id → 向量表示 (d_model)
```

* 输入 `(batch_size, seq_len)`
* 输出 `(batch_size, seq_len, d_model)`

---

## 🔹 RMSNorm

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        # 计算均方根归一化
        rms = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        return x / rms * weight
```

* 与 LayerNorm 不同，**不减均值**，只按 RMS 归一化。
* 数值更稳定，常用于 GPT 系列。

---

## 🔹 SwiGLU

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return W2( SiLU(W1(x)) * W3(x) )
```

* 代替 ReLU 的 **门控激活函数**。
* **优势**：提高表达能力，已用于 LLaMA、PaLM。

---

## 🔹 RoPE (Rotary Positional Embedding)

```python
class ROPE(nn.Module):
    def forward(self, x, token_positions):
        # 将 cos/sin 旋转嵌入到 query/key 向量中
```

* 使用 **正余弦函数**让模型学习序列位置。
* **好处**：支持无限长 extrapolation，比绝对位置编码更灵活。

---

## 🔹 Multi-Head Self Attention

```python
class MultiHeadSelfAttention(nn.Module):
    def forward(self, x, mask=None, token_positions=None):
        q = Wq(x), k = Wk(x), v = Wv(x)
        q, k, v = 头分拆
        q, k = RoPE(q, k)
        attn = softmax(q @ k^T / sqrt(d_k))
        out = attn @ v
```

* 多头注意力，支持 **RoPE**。
* 默认 **causal mask**，保证只看见过去的 token。

---

## 🔹 TransformerBlock

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))
        return x
```

* 残差结构：**前归一化 + 残差连接**。
* 典型的 **Transformer 结构单元**。

---

## 🔹 TransformerLM

```python
class TransformerLM(nn.Module):
    def forward(self, inputs):
        x = Embedding(inputs)
        for block in layers:
            x = block(x)
        logits = lm_head(LN(x))
        return logits
```

* 输入：`(batch_size, seq_len)`
* 输出：`(batch_size, seq_len, vocab_size)`
* 预测每个位置的下一个 token。

---

## 🔹 模型结构示意

```plaintext
输入 tokens → Embedding → [Block × N] → RMSNorm → Linear(vocab_size)
```

其中每个 **Block** 内部是：

```plaintext
x → RMSNorm → MultiHeadSelfAttention → 残差
x → RMSNorm → SwiGLU FeedForward → 残差
```

---

## 🔹 总结

* 本实现属于 **简化版 GPT 架构**：

  * 使用 **RMSNorm + SwiGLU + RoPE** → 更接近 LLaMA 风格。
  * 输出 logits，不在 forward 内部做 softmax，**方便训练时配合 CrossEntropyLoss**。
* 可扩展：增加 `num_layers`、`num_heads`、`d_model` 即可实现更大模型。

---

 