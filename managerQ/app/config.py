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
    qpulse_url: str # URL for the QuantumPulse service

    class Config:
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                lambda s: yaml.safe_load(open("managerQ/config/manager.yaml")),
                env_settings,
                file_secret_settings,
            )

settings = ManagerSettings() 