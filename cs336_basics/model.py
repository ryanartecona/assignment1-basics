import math
from typing import Optional

from jaxtyping import Float
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # TODO: init d_ff to be 8/3 of d_model, rounded to nearest 64 multiple
        # (requires making d_ff optional)
        self.w1 = nn.Parameter(torch.ones([d_ff, d_model], device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.ones([d_model, d_ff], device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.ones([d_ff, d_model], device=device, dtype=dtype))

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
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
def softmax(
    x: Float[torch.Tensor, "..."], dim: Optional[int] = -1
) -> Float[torch.Tensor, "..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum


# Section 3.3.4 - scaled dot product attention
def scaled_dot_product_attention(
    q: Float[torch.Tensor, "... n d_k"],
    k: Float[torch.Tensor, "... m d_k"],
    v: Float[torch.Tensor, "... m d_v"],
    mask: Optional[Float[torch.Tensor, "... n m"]] = None,
) -> Float[torch.Tensor, "... n d_v"]:
    d_k = q.shape[-1]
    inner = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    inner[mask == False] = float("-inf")
    return softmax(inner, dim=-1) @ v
