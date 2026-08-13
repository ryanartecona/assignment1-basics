import math
from typing import Any, Callable, Optional

from einops import rearrange
from jaxtyping import Float, Int
from einops.einops import einsum, reduce, repeat
from torch import nn
import torch


# Section 3.3.2
class Linear(nn.Module):
    w: Float[torch.Tensor, "d_out d_in"]

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # NOTE init by truncated normal distribution not yet verified correct
        w = torch.zeros(
            [d_out, d_in],
            dtype=dtype,
            device=device,
        )
        std = 2.0 / (d_in + d_out)
        nn.init.trunc_normal_(w, mean=0, std=std, a=std * -3, b=std * 3)
        self.w = nn.Parameter(w)

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> torch.Tensor:
        return einsum(self.w, x, "d_out d_in, ... d_in -> ... d_out")


# Section 3.3.3
class Embedding(nn.Module):
    weights: Float[torch.Tensor, "vocab_size d_model"]

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        weights = torch.zeros([vocab_size, d_model], device=device, dtype=dtype)
        # NOTE init by truncated normal distribution not yet verified correct
        nn.init.trunc_normal_(weights, mean=0, std=1, a=-3, b=3)
        self.weights = nn.Parameter(weights)

    def forward(self, token_ids: Float[torch.Tensor, "..."]):
        # treat each token_id as index into embedding matrix
        return self.weights[token_ids]


# Section 3.4.1
class RMSNorm(nn.Module):
    d_model: int
    eps: float
    gain: Float[torch.Tensor, "d_model"]

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        squares = x.square() + self.eps
        means = reduce(squares, "... d_model -> ...", "sum") / self.d_model
        means = repeat(means, "... -> ... d_model", d_model=self.d_model)
        res = (x / means.sqrt()) * self.gain

        return res.to(in_dtype)


# Section 3.4.2
class SwiGLU(nn.Module):
    w1: Float[torch.Tensor, "d_ff d_model"]
    w2: Float[torch.Tensor, "d_model d_ff"]
    w3: Float[torch.Tensor, "d_ff d_model"]

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # TODO: init d_ff to be 8/3 of d_model, rounded to nearest 64 multiple
        # (requires making d_ff optional)
        self.w1 = nn.Parameter(torch.ones([d_ff, d_model], device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.ones([d_model, d_ff], device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.ones([d_ff, d_model], device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # SwiGLU(x, W1, W2, W3) = W2 @ (SiLU(W1 @ x) * W3 @ x)
        # where SiLU(x) = x * sigmoid(x)

        x1 = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        x1_silu = x1 * torch.sigmoid(x1)
        x3 = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")

        # alt 1
        # x_gated = einsum(x1_silu, x3, "... d_ff, ... d_ff -> ... d_ff")
        # alt 2
        x_gated = x1_silu * x3

        res = einsum(self.w2, x_gated, "d_model d_ff, ... d_ff -> ... d_model")
        return res


# Section 3.4.3
class RoPE(nn.Module):
    theta: float
    max_seq_len: int
    d_k: int
    sin_thetas: Float[torch.Tensor, "max_seq_len d_k"]
    cos_thetas: Float[torch.Tensor, "max_seq_len d_k"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_k = d_k

        # prefill 2d cos(theta[i,k]) and sin(theta[i,k]) lookups
        i = torch.arange(max_seq_len)
        k = torch.arange(1, 1 + d_k // 2)
        thetas = torch.outer(i, torch.pow(self.theta, -(2 * k - 2) / d_k))
        sin_thetas = repeat(torch.sin(thetas), "i k -> i (k 2)")
        cos_thetas = repeat(torch.cos(thetas), "i k -> i (k 2)")

        # negate even sins for matmul decomp in forward pass
        evens = torch.arange(self.d_k) % 2 == 0
        sin_mask = torch.ones(self.d_k)
        sin_mask[evens] = -1

        self.register_buffer("sin_thetas", sin_thetas * sin_mask, persistent=False)
        self.register_buffer("cos_thetas", cos_thetas, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Float[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        sin_factor = self.sin_thetas[token_positions]
        cos_factor = self.cos_thetas[token_positions]
        # pairwise roll the last dim of x, i.e. [0,1,2,3,...] -> [1,0,3,2,...]
        x_rot = x.reshape(x.shape[:-1] + (-1, 2)).roll(1, -1).reshape(x.shape)
        return x_rot * sin_factor + x * cos_factor


# Section 3.3.4 - softmax
def softmax(x: Float[torch.Tensor, "..."], dim: int | None = -1) -> Float[torch.Tensor, "..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum


# Section 3.3.4 - scaled dot product attention
def scaled_dot_product_attention(
    q: Float[torch.Tensor, "... n d_k"],
    k: Float[torch.Tensor, "... m d_k"],
    v: Float[torch.Tensor, "... m d_v"],
    mask: Float[torch.Tensor, "... n m"] | None = None,
) -> Float[torch.Tensor, "... n d_v"]:
    d_k = q.shape[-1]
    inner = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    inner[mask == False] = float("-inf")
    return softmax(inner, dim=-1) @ v


# Section 3.4.5 - multihead self attention
class MultiheadSelfAttention(nn.Module):
    n_heads: int
    wQ: Float[torch.Tensor, "h*d_k d_model"]
    wK: Float[torch.Tensor, "h*d_k d_model"]
    wV: Float[torch.Tensor, "h*d_v d_model"]
    wO: Float[torch.Tensor, "d_model h*d_v"]
    rope: RoPE | None

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.rope = rope
        h = n_heads
        d_k = d_model // h
        d_v = d_k
        wQ = torch.zeros([d_k * h, d_model])
        hk_m_stddev = 2.0 / (d_k * h + d_model)
        nn.init.trunc_normal_(wQ, mean=0, std=hk_m_stddev, a=hk_m_stddev * -3, b=hk_m_stddev * 3)
        self.wQ = nn.Parameter(wQ)
        wK = torch.zeros([d_k * h, d_model])
        nn.init.trunc_normal_(wK, mean=0, std=hk_m_stddev, a=hk_m_stddev * -3, b=hk_m_stddev * 3)
        self.wK = nn.Parameter(wK)
        hv_m_stddev = 2.0 / (d_v * h + d_model)
        wV = torch.zeros([d_v * h, d_model])
        nn.init.trunc_normal_(wV, mean=0, std=hv_m_stddev, a=hv_m_stddev * -3, b=hv_m_stddev * 3)
        self.wV = nn.Parameter(wV)
        wO = torch.zeros([d_model, h * d_v])
        nn.init.trunc_normal_(wO, mean=0, std=hv_m_stddev, a=hv_m_stddev * -3, b=hv_m_stddev * 3)
        self.wO = nn.Parameter(wO)

    def _rope(self, qk: Float[torch.Tensor, "... n d_k"]) -> Float[torch.Tensor, "... n d_k"]:
        if self.rope is None:
            return qk
        seq_len = qk.shape[-2]
        token_positions = torch.arange(seq_len, device=qk.device)
        return self.rope(qk, token_positions)

    def forward(self, x: Float[torch.Tensor, "... n d_model"]) -> Float[torch.Tensor, "... n d_model"]:
        mask = torch.ones(x.shape[:-1] + (x.shape[-2],), dtype=torch.bool).tril()
        q_heads = einsum(self.wQ, x, "h_d_k d_model, ... n d_model -> ... h_d_k n")
        k_heads = einsum(self.wK, x, "h_d_k d_model, ... n d_model -> ... h_d_k n")
        v_heads = einsum(self.wV, x, "h_d_v d_model, ... n d_model -> ... h_d_v n")
        res_hbatch = scaled_dot_product_attention(
            q=self._rope(rearrange(q_heads, "... (h d_k) n -> ... h n d_k", h=self.n_heads)),
            k=self._rope(rearrange(k_heads, "... (h d_k) n -> ... h n d_k", h=self.n_heads)),
            v=rearrange(v_heads, "... (h d_v) n -> ... h n d_v", h=self.n_heads),
            mask=repeat(mask, "... n m -> ... h n m", h=self.n_heads),
        )
        res = rearrange(res_hbatch, "... h n d_v -> ... n (h d_v)")
        return einsum(self.wO, res, "d_model h_d_v, ... n h_d_v -> ... n d_model")


# Section 3.5 - transformer block
class TransformerBlock(nn.Module):
    attn_block: nn.Module

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.attn_block = nn.Sequential()
        self.attn_block.add_module("rmsnorm", RMSNorm(d_model))
        self.attn_block.add_module("mhsa", MultiheadSelfAttention(d_model, n_heads, rope=rope))
        self.ff_block = nn.Sequential()
        self.ff_block.add_module("rmsnorm", RMSNorm(d_model))
        self.ff_block.add_module("ffn", SwiGLU(d_model, d_ff))

    def forward(self, x: Float[torch.Tensor, "... n d_model"]) -> Float[torch.Tensor, "... n d_model"]:
        x += self.attn_block(x)
        x += self.ff_block(x)
        return x


# Section 3.5 - transformer model
class TransformerLM(nn.Module):
    token_embeddings: Embedding
    layers: nn.Sequential
    ln_final: RMSNorm
    lm_out: Linear

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        rope_theta: float,
        context_length: int,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_k = d_model // n_heads
        rope = RoPE(theta=rope_theta, d_k=d_k, max_seq_len=context_length)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, n_heads, d_ff, rope=rope) for _ in range(n_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_out = Linear(d_model, vocab_size)

    def forward(self, in_indices: Float[torch.Tensor, "... seq_len"]):
        emb = self.token_embeddings(in_indices)
        res = self.layers(emb)
        out = self.ln_final(res)
        logits = self.lm_out(out)
        return logits


def cross_entropy_loss(
    logits: Float[torch.Tensor, "... seq_len vocab_size"], targets: Int[torch.Tensor, "... seq_len"]
) -> Float[torch.Tensor, "..."]:
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits_adj = logits - logits_max
    logits_exp = torch.exp(logits_adj)
    logits_exp_sum = logits_exp.sum(dim=-1, keepdim=True)
    log_probs = logits_adj - torch.log(logits_exp_sum)
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.99, eps=10e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "b1": betas[0],
            "b2": betas[1],
            "decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> Any:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["b1"]
            b2 = group["b2"]
            decay = group["decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                lr_t = lr * (math.sqrt(1 - b2**t) / (1 - b1**t))

                # in-place weight decay
                p.data -= lr * decay * p.data
                # update moment estimates
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * (grad**2)
                state["m"] = m
                state["v"] = v

                # update weight tensor in-place.
                p.data -= lr_t * (m / (v.sqrt() + eps))
                # increment iteration number
                state["t"] = t + 1
        return loss


# Section 4.4
def lr_schedule_cosine_annealing(t: int, lr_max: float, lr_min: float, warmup_period: int, annealing_period: int):
    '''
    Cosine annealing schedule for learning rate
    '''
    assert warmup_period < annealing_period
    if t < warmup_period:
        return lr_max * t / warmup_period
    elif t <= annealing_period:
        annealed = 0.5 * (1 + (math.cos(math.pi * (t - warmup_period) / (annealing_period - warmup_period))))
        return lr_min + (lr_max - lr_min) * annealed
    else:
        return lr_min
