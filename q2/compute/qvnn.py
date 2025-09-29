from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


class QVNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class TrainResult:
    loss: float
    wall_time_ms: float
    device: str


def fit_qvnn(
    n: int = 4096,
    in_dim: int = 64,
    hidden: int = 128,
    out_dim: int = 1,
    steps: int = 200,
    lr: float = 1e-3,
    amp: bool = True,
    device: torch.device | None = None,
) -> TrainResult:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QVNN(in_dim, hidden, out_dim).to(device)
    x = torch.randn(n, in_dim, device=device)
    w_true = torch.randn(in_dim, out_dim, device=device)
    y = x @ w_true

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    use_amp = amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t0 = time.perf_counter()
    loss_val = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_val = float(loss.detach().item())
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return TrainResult(loss=loss_val, wall_time_ms=dt_ms, device=str(device))