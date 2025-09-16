import json
import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections import defaultdict
import time


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        """
        初始化分词器，包含词表、BPE 合并规则和可选的特殊符号。

        参数:
            vocab (dict[int, bytes]): token ID 到字节编码 token 的映射。
            merges (list[tuple[bytes, bytes]]): BPE 合并操作列表，每个元素是一个字节对。
            special_tokens (list[str] | None): 可选，用户定义的特殊符号。
        """
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}  # 反向映射：bytes → int
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        从词表文件和 merges 文件构造分词器。

        参数:
            vocab_filepath (str): BPE 训练生成的词表文件路径。
            merges_filepath (str): BPE 训练生成的 merges 文件路径。
            special_tokens (list[str] | None): 可选，特殊符号。

        返回:
            Tokenizer: 初始化好的分词器实例。
        """
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split("\t")
                vocab[int(id_str)] = token_str.encode("utf-8")  # 存储为字节

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        使用 BPE 算法将输入字符串编码为 token ID 列表。

        参数:
            text (str): 输入文本。

        返回:
            list[int]: 表示编码结果的 token ID 列表。
        """
        token_ids = []
        pre_tokens_list = process_chunk((text, self.special_tokens, True))
        for tokens in pre_tokens_list:
            for pair in self.merges:
                a, b = pair
                new_tok = a + b
                new_tokens: list[bytes] = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                        new_tokens.append(new_tok)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            for i in range(len(tokens)):
                token_ids.append(self.vocab_reversed.get(tokens[i]))

        return token_ids

    def encode_iterable(self, iterable: list[str]) -> iter:
        """
        按需（惰性）将字符串序列编码为 token ID 流。
        用于大规模数据集的内存高效分词。

        参数:
            iterable (list[str]): 字符串序列（例如文件的每一行）。

        返回:
            iter: 逐个返回 token ID 的生成器。
        """
        for line in iterable:
            token_ids = self.encode(line)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """将 token ID 序列解码为文本。"""
        tokens = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"

        for token_id in ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]
            else:
                token = bytes(replacement_char, encoding='utf-8')  # 越界时替换为 Unicode 占位符

            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    在给定输入语料上训练字节级 BPE 分词器。

    参数
    ----------
    input_path : str
        UTF-8 编码的训练语料文件路径，每一行属于语料的一部分。

    vocab_size : int
        最终词表大小（包含初始字节级 token、训练得到的合并 token，以及特殊符号）。

    special_tokens : list[str]
        用户定义的特殊符号（例如 ["<|endoftext|>", "<pad>"]），不会参与合并。

    num_processes : int, 默认=8
        预分词时使用的并行进程数，每个进程处理一个分块。更多进程通常更快。

    返回
    -------
    vocab : dict[int, bytes]
        token ID (整数) → token（字节）的映射。

    merges : list[tuple[bytes, bytes]]
        BPE 合并操作列表，顺序按合并先后排列。
    """
    # 1. 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    special_tokens = sorted(special_tokens, key=lambda x: -len(x))

    # 2. 预分词
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        chunk_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)
    task_args = [(chunk, special_tokens, False) for chunk in chunk_list]
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args)

    # 3. 计算 BPE 合并
    merges: list[tuple[bytes, bytes]] = []
    pre_tokens_bytes: list[list[bytes]] = [token for chunk in chunk_results for token in chunk]
    counts = defaultdict(int)
    pair_to_indices = defaultdict(set)
    for idx, token in enumerate(pre_tokens_bytes):
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            counts[pair] += 1
            pair_to_indices[pair].add(idx)

    idx = len(vocab)
    while idx < vocab_size:
        if not counts:
            break

        max_pair: tuple[bytes, bytes] = None
        max_cnt = -1
        for pair, cnt in counts.items():
            if cnt > max_cnt:
                max_pair = pair
                max_cnt = cnt
            elif cnt == max_cnt:
                if max_pair is None or pair > max_pair:
                    max_pair = pair

        merges.append(max_pair)
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1

        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                old_pair = (token[i], token[i + 1])
                pair_to_indices[old_pair].discard(j)
                counts[old_pair] -= 1
                if counts[old_pair] == 0:
                    counts.pop(old_pair)
                    pair_to_indices.pop(old_pair, None)

            merged = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == a and token[i + 1] == b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            pre_tokens_bytes[j] = merged

            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                counts[pair] += 1
                pair_to_indices[pair].add(j)

    return vocab, merges


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
) -> list[int]:
    """
    将文件切分为若干块，可独立统计。
    如果边界重叠，返回的分块数可能少于目标值。
    """
    assert isinstance(split_special_token, bytes), "特殊符号必须是字节串"

    # 获取文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 初始分块边界
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向前读取 4KB

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # 到达文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 查找特殊符号
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                true_position = initial_position + found_at
                chunk_boundaries[bi] = true_position
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def process_chunk(args: tuple[str, list[str], bool]) -> list[list[bytes]]:
    chunk, special_tokens, keep_special_tokens = args
    """
    处理一段文本并返回字节级分词结果。

    参数:
        chunk (str): 文本块。
        special_tokens (list[str]): 特殊符号。
        keep_special_tokens (bool): 是否保留特殊符号。

    返回:
        list[list[bytes]]: 分词结果，每个 token 是字节序列。
    """
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if keep_special_tokens and pattern:
        pattern = f"({pattern})"

    segments = re.split(pattern, chunk) if pattern else [chunk]

    pre_tokens_bytes: list[list[bytes]] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for segment in segments:
        if keep_special_tokens and segment in special_tokens:
            # 特殊符号整体保留
            token_bytes = [segment.encode("utf-8")]
            pre_tokens_bytes.append(token_bytes)
        else:
            # 正则分词
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, segment)]
            for token in tokens:
                token_bytes = [bytes([b]) for b in token]
                pre_tokens_bytes.append(token_bytes)

    return pre_tokens_bytes


def save_bpe_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 保存词表
    vocab_serialized = {str(i): token.decode('utf-8', errors='replace') for i, token in vocab.items()}
    with open(os.path.join(output_dir, "TinyStories_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_serialized, f, ensure_ascii=False, indent=2)

    # 保存 merges
    with open(os.path.join(output_dir, "TinyStories_merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            a_str = a.decode("utf-8", errors="replace")
            b_str = b.decode("utf-8", errors="replace")
            f.write(f"{a_str} {b_str}\n")


def main():
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path="data/TinyStories/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    elapsed = time.time() - start_time
    print(f"训练完成，耗时 {elapsed:.2f} 秒")
    print(f"词表大小: {len(vocab)}")
    print(f"最长 token: {max(vocab.values(), key=len)} (长度={len(max(vocab.values(), key=len))})")
    save_bpe_model(vocab, merges, "cs336_basics")


def test():
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    ids = tokenizer.encode(test_string, allowed_special={"<|endoftext|><|endoftext|>", "<|endoftext|>"})
    decoded = [tokenizer.decode([x]) for x in ids]
    print(f"tiktoken 编码结果: {ids}, 解码结果: {decoded}")


if __name__ == "__main__":
    main()
    #test()
