from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional


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

    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    database_url: Optional[str] = None
    db_host: Optional[str] = None
    db_port: int = 5432
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_name: str = "joule_replacement"

    @property
    def database_connection_string(self) -> str:
        if self.database_url:
            return self.database_url
        if self.db_host and self.db_user and self.db_password:
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        return "sqlite:///./data/joule_dev.db"

    # HANA Cloud (optional)
    hana_host: Optional[str] = None
    hana_port: int = 443
    hana_user: Optional[str] = None
    hana_password: Optional[str] = None
    hana_database: Optional[str] = None

    # Vector Store
    vector_store_type: str = "chroma"  # "chroma", "faiss", "pinecone"
    vector_store_path: str = "./data/vector_store"
    vector_store_collection: str = "joule_knowledge"
    embedding_model: str = "text-embedding-3-small"

    # Knowledge Base
    knowledge_base_path: str = "./data/knowledge_base"
    knowledge_base_chunk_size: int = 1000
    knowledge_base_chunk_overlap: int = 200

    # Agent
    max_agent_iterations: int = 10
    agent_timeout_seconds: int = 30

    # Feature Flags
    enable_conversation_memory: bool = True
    enable_multi_agent_orchestration: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()