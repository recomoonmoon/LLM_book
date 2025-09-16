 
---

# Byte-Level BPE Tokenizer

本项目实现了一个 **字节级 BPE（Byte Pair Encoding）分词器训练器**，支持高效的并行预分词与合并操作优化。该实现基于 **Stanford CS336 作业**，并在 TinyStories 数据集上进行了测试。

---

## 📌 功能特性

* **字节级初始化**：以所有 0-255 的字节作为初始词表。
* **特殊符号处理**：支持添加 `<|endoftext|>`、`<pad>` 等用户定义的 special tokens。
* **并行预分词**：

  * 基于文件边界 `<|endoftext|>` 进行切分，避免跨文档。
  * 使用 `multiprocessing.Pool` 加速处理。
* **高效 BPE 合并**：

  * 仅更新受影响的 pair 频率，避免全局统计。
  * 使用 `pair_to_indices` 映射管理 pair → token 的索引集合。
* **最终输出**：

  * `vocab: dict[int, bytes]` — 词表，token ID 到字节的映射。
  * `merges: list[tuple[bytes, bytes]]` — 训练过程中执行的合并操作。

---

## ⚙️ 主要流程

1. **初始化词表**

   ```python
   vocab = {i: bytes([i]) for i in range(256)}
   for tok in special_tokens:
       vocab[len(vocab)] = tok.encode("utf-8")
   ```

2. **并行预分词**

   * 使用 `find_chunk_boundaries` 找到 `<|endoftext|>` 边界。
   * 调用 `process_chunk` 对每个 chunk 分词并转为字节。

3. **BPE 训练**

   * 统计 pair 出现频率 `counts`。
   * 选择频率最高的 pair `(a, b)`，执行合并并更新 `vocab`。
   * 更新受影响 token 的 pair 计数。
   * 重复，直到 `vocab_size` 达到目标。

---

## 📂 核心模块

* `train_bpe` — 主入口函数
* `find_chunk_boundaries` — 数据集切分
* `process_chunk` — 预分词
* **输出**：`vocab`, `merges`

---

## 🧪 测试方法

```bash
uv run pytest tests/test_train_bpe.py
```

确保实现能通过三个测试用例：

* 词表构造
* 特殊 token 行为
* 合并顺序正确性

---
## 作业：
 * 完成本目录下test.py的代码，实现Tokenizer（完整版Tokenizer.py）。
 * 如果有余力，可以实现其训练过程

---

## 📖 参考

* Stanford CS336 作业说明
* [Sennrich et al. (2016) - Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
* HuggingFace `tokenizers` 源码

---

 