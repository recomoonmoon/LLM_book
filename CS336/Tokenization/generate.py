import torch
import torch.nn.functional as F

@torch.no_grad()  # 禁止梯度计算，加快推理速度
def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,   # 最大生成 token 数
    temperature: float = 1.0,   # 温度系数（>1 更随机，<1 更确定）
    top_p: float = 1.0,         # nucleus sampling 阈值，控制多样性
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> str:
    """
    使用给定模型和 tokenizer，根据输入提示（prompt）生成文本。

    参数：
        model: 训练好的 TransformerLM 模型。
        tokenizer: 具备 encode() 和 decode() 方法的分词器。
        prompt (str): 起始文本（条件上下文）。
        max_new_tokens (int): 最大生成的新 token 数。
        temperature (float): softmax 温度采样参数。
        top_p (float): nucleus sampling（核采样）概率阈值，控制保留概率累计不超过 p 的候选。
        device (str): 推理所用设备（GPU/CPU）。

    返回：
        str: 包含 prompt 和生成内容的最终字符串。
    """
    model.eval()
    model.to(device)

    # 将 prompt 编码为 token IDs，并放到 device 上
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 预先缓存 "<|endoftext|>" 的 ID，避免循环中重复 encode
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    for _ in range(max_new_tokens):
        # 前向传播，得到所有位置的 logits
        logits = model(input_ids)        # (1, seq_len, vocab_size)
        logits = logits[:, -1, :]        # 仅取最后一个位置的 logits (1, vocab_size)

        # 温度缩放
        logits = logits / max(temperature, 1e-8)

        # softmax 转为概率分布
        probs = F.softmax(logits, dim=-1)

        # Top-p (nucleus) 采样
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 筛掉累计概率超过 top_p 的部分
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()  # 保证至少保留第一个 token
        cutoff[..., 0] = False

        # 将超过 top_p 的概率置零
        sorted_probs[cutoff] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # 从保留的分布中采样下一个 token
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_token)  # 映射回原始 token id

        # 拼接到当前序列
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 如果生成了 EOS，提前停止
        if next_token.item() == eos_id:
            break

    # 解码为字符串
    return tokenizer.decode(input_ids[0].tolist())
