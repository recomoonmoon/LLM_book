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
        初始化分词器，保存词表、BPE 合并规则和特殊符号。

        参数:
            vocab (dict[int, bytes]): token ID 到字节 token 的映射。
            merges (list[tuple[bytes, bytes]]): BPE 合并规则，每个元素是两个字节 token 的元组。
            special_tokens (list[str] | None): 可选的特殊符号列表。

        要做的事:
            - 保存词表到 self.vocab
            - 建立反向词表 (bytes → int)，保存到 self.vocab_reversed
            - 保存 merges 规则
            - 将特殊符号按照长度从长到短排序，保存到 self.special_tokens
        """


        self.vocab = vocab
        self.merges = merges
        self.vocab_reversed = {v:k for k,v in self.vocab.items()}
        #优先处理长特殊符号，以免截断（先处理短的然后把长的截取掉了）
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))

    @classmethod
    def from_files(self, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        从文件中加载词表和 merges 规则，构造一个 Tokenizer 实例。

        参数:
            vocab_filepath (str): 词表文件路径，每行格式为 "id<TAB>token"
            merges_filepath (str): merges 文件路径，每行两个 token 表示一条合并规则
            special_tokens (list[str] | None): 可选的特殊符号列表

        返回:
            Tokenizer: 初始化好的分词器对象

        要做的事:
            - 读取 vocab 文件，建立 {id: token} 字典（注意 token 存为 bytes 类型）
            - 读取 merges 文件，建立 [(a, b), ...] 的合并规则列表（元素是 bytes 对）
            - 调用 cls(vocab, merges, special_tokens) 返回一个分词器实例
        """
        vocab_file = open(vocab_filepath, 'r', encoding='utf-8')
        for line in vocab_file.readlines():
            id_str, token_str = line.strip().split("\t")
            self.vocab[int(id_str)] = token_str.encode("utf-8")

        merge_file = open(merges_filepath, 'r', encoding='utf-8')
        for line in merge_file:




    def encode(self, text: str) -> list[int]:
        """
        将字符串编码为 token ID 列表。

        参数:
            text (str): 输入字符串

        返回:
            list[int]: 对应的 token ID 序列

        要做的事:
            - 先对 text 进行切分，识别出普通字符和特殊符号
            - 依次应用 BPE merges 规则，把相邻 token 合并
            - 将最终的 bytes token 映射为 ID（用 vocab_reversed 查表）
            - 返回 token ID 列表
        """
        pass

    def encode_iterable(self, iterable: list[str]) -> iter:
        """
        按需编码一个字符串序列，返回一个迭代器（生成器）。

        参数:
            iterable (list[str]): 字符串序列，例如一行行文本

        返回:
            iter: 逐个产生 token ID 的生成器

        要做的事:
            - 遍历 iterable 中的每个字符串
            - 对每个字符串调用 encode
            - 使用 yield from 逐个返回 token ID（节省内存）
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 序列解码为原始字符串。

        参数:
            ids (list[int]): token ID 序列

        返回:
            str: 解码后的字符串

        要做的事:
            - 遍历 token ID 列表
            - 将每个 ID 转换为对应的 bytes token（查 vocab）
            - 拼接所有 token 并 decode 为字符串
            - 对超出 vocab 范围的 ID，用 Unicode 占位符替换
        """
        pass
