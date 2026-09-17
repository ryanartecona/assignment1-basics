from typing import cast

import numpy as np
import pytest
import torch

from cs336_basics.model import AdamW, TransformerLM, save_checkpoint


def test_torch_load_transformer(tmp_path):
    model1 = TransformerLM(10, 11, 12, 13, 14, 15, 16)
    optimizer = AdamW(model1.parameters())
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(model1.state_dict(), checkpoint_path)
    loaded = torch.load(checkpoint_path)
    model2 = TransformerLM.from_state_dict(loaded)
    model1_state_shape = {k: cast(torch.Tensor, v).shape for k, v in model1.state_dict().items()}
    model2_state_shape = {k: cast(torch.Tensor, v).shape for k, v in model2.state_dict().items()}
    assert model1_state_shape == model2_state_shape
    
