# AuthQ/app/main.py
from fastapi import FastAPI
import uvicorn
import structlog
from contextlib import asynccontextmanager
from keycloak import KeycloakAdmin, KeycloakOpenID

from app.api import auth, users
from app.core.config import settings
from shared.vault_client import VaultClient
from shared.observability.logging_config import setup_logging
from shared.observability.metrics import setup_metrics

# --- Logging ---
setup_logging(service_name=settings.service_name)
logger = structlog.get_logger(__name__)

def load_config_from_vault():
    """Loads configuration from Vault and populates the settings object."""
    try:
        vault_client = VaultClient(role="authq-role")
        config = vault_client.read_secret_data("secret/data/authq/config")
        if not config:
            raise ValueError("AuthQ config not found in Vault.")
        
        # Populate the settings object
        settings.keycloak = config['keycloak']
        
    except Exception as e:
        logger.critical(f"Failed to load configuration from Vault: {e}", exc_info=True)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the Keycloak clients.
    """
    logger.info("AuthQ service starting up...")
    load_config_from_vault()
    
    # Initialize Keycloak clients and store them in app state
    app.state.keycloak_openid = KeycloakOpenID(
        server_url=settings.keycloak.server_url,
        client_id=settings.keycloak.client_id,
        realm_name=settings.keycloak.realm_name,
        client_secret_key=settings.keycloak.client_secret
    )
    app.state.keycloak_admin = KeycloakAdmin(
        server_url=settings.keycloak.server_url,
        username=settings.keycloak.admin_username,
        password=settings.keycloak.admin_password,
        realm_name=settings.keycloak.realm_name,
        client_id=settings.keycloak.client_id,
        client_secret_key=settings.keycloak.client_secret,
        verify=True
    )
    yield
    logger.info("AuthQ service shutting down...")
    # No explicit close needed for these clients


# --- FastAPI App ---
app = FastAPI(
    title="AuthQ",
    description="Authentication and Authorization Service for the Q Platform.",
    version="0.2.0",
    lifespan=lifespan
)

# --- API Routers ---
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])


@app.get("/health", tags=["Health"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    host = os.getenv("API_HOST", "127.0.0.1")  # Default to localhost instead of all interfaces
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
