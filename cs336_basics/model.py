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
        device: torch.device = None,
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
        device: torch.device = None,
        dtype: torch.dtype = None,
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
        device: torch.device = None,
        dtype: torch.dtype = None,
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
        device: torch.device = None,
        dtype: torch.dtype = None,
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
