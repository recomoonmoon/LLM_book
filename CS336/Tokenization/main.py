#2.1 Unicode 标准
import re

print(ord("牛"))
print(chr(29275))

"""
unicode1
(a) chr(0) 返回什么字符？ 
(b) 它的 __repr__() 和打印结果有什么区别？ 
(c) 当该字符出现在字符串中会发生什么？
"""

print(chr(0))
#（a）输出命令行看不清楚的“ ”，属于占位符or空格？

print(chr(0).__repr__())
#输出 '\x00' 应该是十六进制的表述

print("11"+chr(0)+"11")
#terminal输出有特殊的一个符号，复制后变成空格

"""---------------------------------------------------------------------------------------"""
#2.2 Unicode 编码

#编码和解码
text = "Make America Great Again"
print(text)
print(text.encode('utf-8'))
print(text.encode('utf-8').decode("utf-8"))

#查看0-255字节值 输入需要迭代对象，比如encode的结果
print(list(text.encode('utf-8')))

"""
作业题 (unicode2, 3分) 
(a) 为什么我们更倾向于用 UTF-8 字节 训练分词器，而不是 UTF-16 或 UTF-32？ 
(b) 下列函数为什么错误？给一个例子。
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
(c) 给出一个不能解码为 Unicode 字符的 2 字节序列。
"""

#(a) 第一utf-8长度更短，其次常用的字符更多，避免稀疏带来的不均衡问题
#(b) 一个utf-8字符编码后长度不一定为1个字节，所以[bytes([b])就是把一个字节当成了整个字符进行decode出错

def test_2_c():
    for i in range(256):
        for j in range(256):
            try:
                ans = bytes([i, j]).decode('utf-8')
            except:
                print(f"{i} {j}")

#(c) 一个字节是8位，0-255，实测发现如果其中一个整数在128-255就不能解码为 Unicode 字符


"""---------------------------------------------------------------------------------------"""






