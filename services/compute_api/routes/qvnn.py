from __future__ import annotations

from fastapi import APIRouter
import torch

from q2.compute.device import get_device
from q2.compute.qvnn import fit_qvnn
from q2.data.api import QvnnTrainRequest, QvnnTrainResponse

router = APIRouter(tags=["qvnn"])


@router.post("/qvnn/train", response_model=QvnnTrainResponse)
def train_qvnn(req: QvnnTrainRequest) -> QvnnTrainResponse:
    dev = get_device()
    res = fit_qvnn(
        n=req.n,
        in_dim=req.in_dim,
        hidden=req.hidden,
        out_dim=req.out_dim,
        steps=req.steps,
        lr=req.lr,
        amp=req.amp,
        device=dev,
    )
    return QvnnTrainResponse(loss=res.loss, wall_time_ms=res.wall_time_ms, device=res.device)