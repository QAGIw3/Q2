import logging
import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

from shared.vault_client import VaultClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Configuration ---


class PulsarTopics(BaseModel):
    requests: str
    preprocessed: str
    routed_prefix: str
    results: str
    feedback: str
    analytics: str
    model_updates: str


class PulsarConfig(BaseModel):
    service_url: str
    tls_trust_certs_file_path: Optional[str] = None
    token: Optional[str] = None
    topics: PulsarTopics


class ApiConfig(BaseModel):
    host: str
    port: int


class IgniteConfig(BaseModel):
    addresses: List[str]
    cluster_name: str
    cache_name: str


class ModelShardConfig(BaseModel):
    name: str = Field(..., alias="name")
    shards: List[str]


class FlinkConfig(BaseModel):
    rest_url: str
    prompt_optimizer_jar_path: str
    dynamic_router_jar_path: str


class OtelConfig(BaseModel):
    """Configuration for OpenTelemetry."""

    enabled: bool = True
    endpoint: Optional[str] = "http://localhost:4317"  # OTLP gRPC endpoint


class AppConfig(BaseModel):
    """The main configuration model for the application."""

    service_name: str
    version: str
    pulsar: PulsarConfig
    api: ApiConfig
    ignite: IgniteConfig
    models: List[ModelShardConfig]
    flink: FlinkConfig
    otel: OtelConfig = Field(default_factory=OtelConfig)


# --- Configuration Loading ---

_config: Optional[AppConfig] = None


def _load_from_file(path: str) -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)


def load_config() -> AppConfig:
    """Loads configuration from Vault, falling back to a local file for dev.

    Fallback order:
      1. Vault (if VAULT_TOKEN / k8s auth present)
      2. File specified by QUANTUMPULSE_CONFIG_FILE
      3. Default local file at QuantumPulse/app/config/local_config.yaml
    """
    global _config
    if _config:
        return _config

    # Allow explicit skip
    skip_vault = os.environ.get("QUANTUMPULSE_SKIP_VAULT", "0") == "1"
    vault_errors = []
    if not skip_vault:
        try:
            logger.info("Attempting to load QuantumPulse configuration from Vault...")
            vault_client = VaultClient(role="quantumpulse-role")
            config_data = vault_client.read_secret_data("secret/data/quantumpulse/config")
            _config = AppConfig(**config_data)
            logger.info("QuantumPulse configuration loaded from Vault.")
            return _config
        except Exception as e:
            vault_errors.append(str(e))
            logger.warning("Vault config load failed. Falling back to local file. Error: %s", e)

    local_path = os.environ.get("QUANTUMPULSE_CONFIG_FILE") or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "local_config.yaml"
    )
    if os.path.exists(local_path):
        try:
            _config = _load_from_file(local_path)
            logger.info("Loaded QuantumPulse configuration from local file: %s", local_path)
            if vault_errors:
                logger.debug("Previous Vault errors: %s", vault_errors)
            return _config
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Failed parsing local QuantumPulse config file %s: %s", local_path, e, exc_info=True)
            raise
    else:
        logger.critical("No configuration source available (Vault failed & local file missing: %s)", local_path)
        raise FileNotFoundError(
            f"QuantumPulse configuration not found. Checked Vault (errors={vault_errors}) and {local_path}"
        )


def get_config() -> AppConfig:
    """
    Dependency injector style function to get the loaded configuration.
    """
    if not _config:
        return load_config()
    return _config


# Load the configuration on module import
config = get_config()
