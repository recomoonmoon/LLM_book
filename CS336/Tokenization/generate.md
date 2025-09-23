 
---

# 文本生成解码函数原理与实现

## 目录

1. [背景与原理](#背景与原理)
   * 1.1 [自回归语言模型](#自回归语言模型)
   * 1.2 [采样方法](#采样方法)
2. [函数设计](#函数设计)
   * 2.1 [函数参数](#函数参数)
   * 2.2 [解码流程](#解码流程)
3. [核心技术](#核心技术)
   * 3.1 [温度采样 (Temperature Sampling)](#温度采样-temperature-sampling)
   * 3.2 [核采样 (Nucleus Sampling / Top-p)](#核采样-nucleus-sampling--top-p)
   * 3.3 [提前终止条件](#提前终止条件)
4. [代码逐行讲解](#代码逐行讲解)
5. [总结](#总结)

---

## 背景与原理

### 1.1 自回归语言模型

Transformer 语言模型是一种 **自回归模型**：

* 它在每个时间步预测 **下一个 token 的概率分布**：

$$
P(x_t | x_{<t})
$$

* 解码时，将前面已生成的 token 作为输入，预测下一个 token，循环迭代直到满足条件（长度/EOS）。

因此，文本生成的关键问题是：**如何从概率分布中选择下一个 token？**

---

### 1.2 采样方法

常见策略：

1. **贪心搜索 (Greedy Search)**

   * 选择概率最大的 token。
   * 缺点：文本缺乏多样性，可能陷入重复循环。

2. **随机采样 (Random Sampling)**

   * 按照概率分布直接采样。
   * 缺点：可能采到罕见词，导致胡言乱语。

3. **温度采样 (Temperature Sampling)**

   * 调整概率分布的“尖锐/平滑程度”。

4. **Top-k / Top-p (Nucleus Sampling)**

   * 限制候选集，使生成结果既合理又多样。

本函数采用 **温度采样 + 核采样（Top-p）**。

---

## 函数设计

### 2.1 函数参数

```python
def decode(model, tokenizer, prompt: str,
           max_new_tokens: int = 50,
           temperature: float = 1.0,
           top_p: float = 1.0,
           device: str = 'cuda' if torch.cuda.is_available() else 'cpu')
```

* **model**：训练好的 Transformer 语言模型。
* **tokenizer**：分词器，支持 `encode` 与 `decode`。
* **prompt**：输入提示，作为生成的上下文条件。
* **max\_new\_tokens**：最多生成多少新 token。
* **temperature**：温度参数，控制随机性。
* **top\_p**：核采样阈值，限制候选 token。
* **device**：运行设备（CPU/GPU）。

---

### 2.2 解码流程

1. 将 `prompt` 转换为 **token IDs**。
2. 逐步预测下一个 token：

   * 得到 **logits（未归一化分数）**。
   * 应用 **温度缩放**。
   * 转为概率分布，并执行 **核采样**。
3. 将采样的 token 拼接到序列中。
4. 如果遇到 **结束符 `<|endoftext|>`**，则提前停止。
5. 最终解码为字符串输出。

---

## 核心技术

### 3.1 温度采样 (Temperature Sampling)

公式：

$$
P_i = \frac{\exp(\frac{z_i}{T})}{\sum_j \exp(\frac{z_j}{T})}
$$

其中：

* $z_i$：logits
* $T$：温度

  * $T < 1$：分布更尖锐 → 更确定
  * $T > 1$：分布更平滑 → 更随机

代码实现：

```python
logits = logits / max(temperature, 1e-8)
```

---

### 3.2 核采样 (Nucleus Sampling / Top-p)

* **思想**：不固定候选数，而是选取最小集合，使其概率和 ≥ `p`。
* 避免只取前 k 个 token（Top-k）时过于死板。

实现步骤：

1. 排序概率：`sorted_probs, sorted_indices = torch.sort(probs, descending=True)`
2. 计算累计和：`cumulative_probs = torch.cumsum(sorted_probs, dim=-1)`
3. 去掉累计超过 `top_p` 的 token。
4. 在保留的分布中重新归一化，并采样。

---

### 3.3 提前终止条件

* 当生成 `<|endoftext|>`（EOS token）时，立即停止。
* 避免无意义的超长生成。

---

## 代码逐行讲解

```python
@torch.no_grad()  
def decode(model, tokenizer, prompt, ...):
    model.eval()
    model.to(device)
```

➡ 禁止梯度，设置模型为推理模式。

```python
input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
eos_id = tokenizer.encode("<|endoftext|>")[0]
```

➡ 编码输入，并取出 `<|endoftext|>` 的 ID。

```python
for _ in range(max_new_tokens):
    logits = model(input_ids)        # 前向传播
    logits = logits[:, -1, :]        # 取最后一个位置
    logits = logits / max(temperature, 1e-8)
    probs = F.softmax(logits, dim=-1)
```

➡ 得到下一个 token 的概率分布。

```python
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

cutoff = cumulative_probs > top_p
cutoff[..., 1:] = cutoff[..., :-1].clone()
cutoff[..., 0] = False
sorted_probs[cutoff] = 0
sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
```

➡ 执行 **核采样**，只保留累计概率 ≤ `top_p` 的 token。

```python
next_token = torch.multinomial(sorted_probs, num_samples=1)
next_token = sorted_indices.gather(-1, next_token)
input_ids = torch.cat([input_ids, next_token], dim=-1)

if next_token.item() == eos_id:
    break
```

➡ 采样并拼接 token，若遇到 EOS 提前停止。

```python
return tokenizer.decode(input_ids[0].tolist())
```

➡ 最终解码为字符串。

---

## 总结

本解码函数实现了一个 **基于 Transformer 的文本生成器**，其关键要点：

* 使用 **自回归生成**，逐步预测下一个 token。
* 通过 **温度采样** 调节随机性。
* 结合 **核采样 (top-p)** 保证生成的合理性与多样性。
* 支持 **EOS 提前停止**，提升效率。
* 最终输出自然语言字符串。

这种方法是 GPT、LLaMA 等大型语言模型生成文本的核心机制。

---
 