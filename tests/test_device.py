from __future__ import annotations

import os

from q2.compute.device import get_device


def test_get_device_cpu_env(monkeypatch):
    monkeypatch.setenv("Q2_DEVICE", "cpu")
    dev = get_device()
    assert str(dev) == "cpu"