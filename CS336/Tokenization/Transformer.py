import torch
from torch import nn
from einops import einsum, rearrange
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        线性层 (无偏置)，执行 y = x @ W^T
        Args:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        # 权重矩阵 (out_features, in_features)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # 截断正态分布初始化
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        """
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Embedding 层
        Args:
            num_embeddings: 词表大小
            embedding_dim: 词向量维度 (d_model)
        """
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((self.vocab_size, self.d_model), **factory_kwargs))
        # 截断正态分布初始化
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, seq_len)
        Returns:
            (batch_size, seq_len, d_model)
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Root Mean Square LayerNorm
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(self.d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            同形状的归一化输出
        """
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)  # 提升到 FP32 提高数值稳定性
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        out = x / rms * self.weight
        return out.to(dtype=in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        SwiGLU 前馈网络
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GLU 变体：SiLU(w1(x)) * w3(x)
        return self.w2(self._silu(self.w1(x)) * self.w3(x))


class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Rotary Position Embedding
        Args:
            theta: 基频超参数
            d_k: 每个 head 的维度 (必须为偶数)
            max_seq_len: 支持的最大序列长度
        """
        super().__init__()
        freqs_d = 1 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos_i = torch.arange(max_seq_len, device=device).float()
        freqs = einsum(freqs_d, pos_i, "d_half, max_seq_len -> max_seq_len d_half")
        cos, sin = torch.cos(freqs), torch.sin(freqs)

        # 缓存 cos/sin
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)
        """
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        cos, sin = self.cos_cached[token_positions], self.sin_cached[token_positions]

        out1 = cos * x_even - sin * x_odd
        out2 = sin * x_even + cos * x_odd
        return torch.stack([out1, out2], dim=-1).flatten(-2)


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    标准缩放点积注意力
    Args:
        query: (..., seq_len_q, d_k)
        key:   (..., seq_len_k, d_k)
        value: (..., seq_len_k, d_v)
        mask:  (..., seq_len_q, seq_len_k) True=可见, False=mask
    """
    d_k = query.shape[-1]
    attention = einsum(query, key, "... q d_k, ... k d_k -> ... q k") / (d_k ** 0.5)
    if mask is not None:
        attention = attention.masked_fill(~mask, float('-inf'))
    attn = F.softmax(attention, dim=-1)
    return einsum(attn, value, "... q k, ... k d_v -> ... q d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta=None, max_seq_len=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        self.output_proj = Linear(num_heads * self.d_v, d_model)

        self.rope = ROPE(theta, self.d_k, max_seq_len) if theta and max_seq_len else None

    def forward(self, x: torch.Tensor, mask=None, token_positions=None) -> torch.Tensor:
        *b, seq_len, _ = x.shape
        q = rearrange(self.q_proj(x), "... n (h d) -> ... h n d", h=self.num_heads, d=self.d_k)
        k = rearrange(self.k_proj(x), "... n (h d) -> ... h n d", h=self.num_heads, d=self.d_k)
        v = rearrange(self.v_proj(x), "... n (h d) -> ... h n d", h=self.num_heads, d=self.d_v)

        # 应用 RoPE
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                for _ in range(len(b)):
                    token_positions = token_positions.unsqueeze(0)
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)

        # 默认 causal mask
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            mask = mask.expand(*b, seq_len, seq_len)

        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None, token_positions=None):
        x = x + self.attn(self.ln1(x), mask, token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, seq_len)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        x = self.token_embeddings(inputs)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
