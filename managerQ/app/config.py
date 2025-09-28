import os
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class PulsarTopics(BaseModel):
    registration: str
    results: str
    platform_events: str
    task_prefix: str


class PulsarConfig(BaseModel):
    service_url: str
    topics: PulsarTopics


class IgniteConfig(BaseModel):
    addresses: list[str]


class ApiConfig(BaseModel):
    host: str
    port: int


class ManagerSettings(BaseSettings):
    service_name: str
    version: str
    pulsar: PulsarConfig
    ignite: IgniteConfig
    api: ApiConfig
    qpulse_url: str  # URL for the QuantumPulse service

    class Config:
        env_prefix = "MANAGERQ_"


def _load_local_config() -> dict:
    """Deterministically load local configuration with sane defaults.

    Order of precedence (highest first):
      1. Environment variables (handled by pydantic after we build base dict)
      2. local_config.yaml (developer provided overrides)
      3. Built-in defaults (ensuring required keys exist)
    """
    base_dir = Path(__file__).resolve().parents[1]  # .../managerQ
    local_cfg = base_dir / "config" / "local_config.yaml"

    data: dict = {}
    if local_cfg.exists():
        try:
            with open(local_cfg, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}

    # Map legacy nested quantumpulse.url -> qpulse_url
    if "qpulse_url" not in data:
        qpulse = data.get("quantumpulse", {})
        if isinstance(qpulse, dict) and "url" in qpulse:
            data["qpulse_url"] = qpulse["url"]

    # Defaults
    data.setdefault("service_name", "managerQ")
    data.setdefault("version", "0.1.0")
    pulsar_dict = data.setdefault("pulsar", {})
    pulsar_dict.setdefault("service_url", "pulsar://localhost:6650")
    topics = pulsar_dict.setdefault("topics", {})
    topics.setdefault("registration", "persistent://public/default/agent-registration")
    topics.setdefault("results", "persistent://public/default/task-results")
    topics.setdefault("task_prefix", "persistent://public/default/tasks-")
    topics.setdefault("platform_events", topics.get("platform_events", "persistent://public/default/platform-events"))
    data.setdefault("ignite", {"addresses": ["127.0.0.1:10800"]})
    data.setdefault("api", {"host": "0.0.0.0", "port": 8001})
    data.setdefault("qpulse_url", "http://localhost:8010")

    # Strip unsupported keys so validation is clean
    for extraneous in ("vectorstore_q", "knowledgegraph_q", "quantumpulse"):
        data.pop(extraneous, None)

    return data


_base_data = _load_local_config()
# Environment variables can still override due to BaseSettings behavior; we just supply base via **
settings = ManagerSettings(**_base_data)
