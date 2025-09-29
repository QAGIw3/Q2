from __future__ import annotations

import os
import random
from typing import Final

import numpy as np
import torch

SEED_DEFAULT: Final[int] = 1337


def seed_all(seed: int = SEED_DEFAULT) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    choice = os.getenv("Q2_DEVICE", "auto").lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_setup() -> None:
    torch.set_float32_matmul_precision("medium")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
