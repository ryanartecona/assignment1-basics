from typing import Optional

from jaxtyping import Float
from einops.einops import einsum
from torch import nn
import torch


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        # NOTE init by truncated normal distribution not yet verified correct
        W = torch.zeros(
            [self.out_features, self.in_features],
            dtype=self.dtype,
            device=self.device,
        )
        std = 2.0 / (in_features + out_features)
        nn.init.trunc_normal_(W, mean=0, std=std, a=std * -3, b=std * 3)
        self.W = nn.Parameter(W)

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
