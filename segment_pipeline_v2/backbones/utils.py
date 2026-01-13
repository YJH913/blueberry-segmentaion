from typing import Optional

import torch
from torch import nn


def load_state_dict(model: nn.Module, weights_path: Optional[str]) -> None:
    if not weights_path:
        return
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
