from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


def _read_xsuaa_from_vcap() -> dict:
    """Read XSUAA credentials from CF VCAP_SERVICES environment variable."""
    vcap_str = os.environ.get("VCAP_SERVICES")
    if not vcap_str:
        return {}
    try:
        vcap = json.loads(vcap_str)
        xsuaa_list = vcap.get("xsuaa", [])
        if not xsuaa_list:
            return {}
        creds = xsuaa_list[0].get("credentials", {})
        result = {}
        if creds.get("verificationkey"):
            result["xsuaa_public_key"] = creds["verificationkey"]
        if creds.get("url"):
            result["xsuaa_issuer"] = creds["url"]
        logger.info(f"Loaded XSUAA config from VCAP_SERVICES (issuer: {result.get('xsuaa_issuer')})")
        return result
    except Exception as e:
        logger.warning(f"Failed to parse VCAP_SERVICES: {e}")
        return {}


class Settings(BaseSettings):
    app_name: str = "Joule Replacement"
    app_version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    use_mock_agent: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2048

    llm_provider: str = "openai"
    sap_aicore_url: Optional[str] = None
    sap_aicore_auth_url: Optional[str] = None
    sap_aicore_client_id: Optional[str] = None
    sap_aicore_client_secret: Optional[str] = None
    sap_aicore_model_id: str = "gpt-4o"
    sap_aicore_deployment_id: str = "default"
    # Separate deployment for text-embedding model (e.g. text-embedding-3-small)
    # If not set, falls back to sap_aicore_deployment_id
    sap_aicore_embedding_deployment_id: Optional[str] = None
    # Resource group for SAP AI Core inference calls (AI-Resource-Group header).
    # Check AI Launchpad → your AI API instance → Resource Groups for the correct value.
    sap_aicore_resource_group: str = "default"

    jwt_secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    token_refresh_days: int = 7

    xsuaa_public_key: Optional[str] = None
    xsuaa_issuer: Optional[str] = None

    @property
    def xsuaa_public_key_formatted(self) -> Optional[str]:
        """Convert escaped newlines in public key to actual newlines"""
        if not self.xsuaa_public_key:
            return None
        return self.xsuaa_public_key.replace('\\n', '\n')

    allowed_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost:4004"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    neon_db_url: Optional[str] = None

    @property
    def async_database_url(self) -> Optional[str]:
        """SQLAlchemy async URL for the Tool Registry's psycopg-v3 engine.

        Derived from ``neon_db_url`` by swapping the scheme to
        ``postgresql+psycopg://`` (psycopg v3 async). The raw ``neon_db_url`` is
        left untouched so the existing sync psycopg2 code keeps using it verbatim.
        psycopg v3 forwards libpq params (sslmode, channel_binding) to libpq, so
        the Neon pooler URL works as-is — no query-string surgery needed.
        Returns None when no Neon URL is configured (registry degrades gracefully).
        """
        raw = self.neon_db_url
        if not raw:
            return None
        if raw.startswith("postgresql+"):  # already carries an explicit driver
            return raw
        if raw.startswith("postgresql://"):
            return "postgresql+psycopg://" + raw[len("postgresql://"):]
        if raw.startswith("postgres://"):
            return "postgresql+psycopg://" + raw[len("postgres://"):]
        return raw

    # Vector Store
    vector_store_collection: str = "joule_knowledge"
    embedding_model: str = "text-embedding-3-small"

    # Knowledge Base
    knowledge_base_path: str = "./data/knowledge_base"
    knowledge_base_chunk_size: int = 1000
    knowledge_base_chunk_overlap: int = 200

    # Live data display & export
    # Number of rows shown inline in the chat response.
    # When total_count > this, download links are offered instead.
    odata_display_rows: int = 10
    # Maximum rows fetched from OData for inline display (must be >= odata_display_rows)
    odata_fetch_rows: int = 50
    # Maximum rows fetched when doing a group-by aggregation
    odata_aggregate_rows: int = 500
    # Base URL used when building export download links in responses.
    # Leave empty (default) to auto-detect from the incoming HTTP request.
    # Set via BACKEND_BASE_URL env var when behind a reverse proxy or in production.
    backend_base_url: str = ""
    # Fallback base URL for CAP app when not registered via cap-plugin.
    # In production set this to your CAP deployment URL, e.g. https://my-cap-app.cfapps.eu10.hana.ondemand.com
    # The registry (app_base_url from cap-plugin) always takes precedence over this value.
    cap_app_base_url: str = ""
    # OData query cache TTL in minutes (set to 0 to disable caching)
    query_cache_ttl_minutes: int = 30

    # Feature Flags
    # When True, chat turns build their context via the new
    # Planner → Retrieval Orchestrator → Context Builder pipeline before the LLM
    # call. Default False = unchanged legacy flow (each agent does its own
    # retrieval). Env: ENABLE_CONTEXT_PIPELINE.
    enable_context_pipeline: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    vcap_overrides = _read_xsuaa_from_vcap()
    return Settings(**vcap_overrides)