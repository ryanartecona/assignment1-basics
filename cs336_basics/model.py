from typing import Optional

from jaxtyping import Float
from einops.einops import einsum
from torch import nn
import torch


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
