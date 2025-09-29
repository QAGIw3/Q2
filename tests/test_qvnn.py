from __future__ import annotations

import torch

from q2.compute.device import get_device, seed_all, torch_setup
from q2.compute.qvnn import fit_qvnn


def test_fit_qvnn_runs_small():
    seed_all(42)
    torch_setup()
    dev = get_device()
    res = fit_qvnn(n=64, in_dim=8, hidden=16, out_dim=1, steps=5, lr=1e-2, amp=False, device=dev)
    assert isinstance(res.loss, float)
    assert res.wall_time_ms >= 0.0
    assert res.device in {"cpu", "cuda:0", "cuda"}