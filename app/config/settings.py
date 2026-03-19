"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application
    app_name: str = "Joule Replacement"
    app_version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    use_mock_agent: bool = False  # Set to True to use mock agent instead of OpenAI (for testing/demo)
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # LLM Configuration
    # REQUIRED: Must be set in environment
    openai_api_key: Optional[str] = None  # Only required if using OpenAI provider
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2048
    
    # SAP AI Core Configuration (Read from environment only)
    # REQUIRED if llm_provider=sap_ai_core: Must be set in .env
    llm_provider: str = "openai"  # Options: "openai", "sap_ai_core", "mock"
    sap_aicore_url: Optional[str] = None
    sap_aicore_auth_url: Optional[str] = None  # OAuth2 authentication URL (from service key "url" field)
    sap_aicore_client_id: Optional[str] = None
    sap_aicore_client_secret: Optional[str] = None
    sap_aicore_model_id: str = "llama-2-7b"
    sap_aicore_deployment_id: str = "default"
    
    # Alternative LLM providers (Read from environment only)
    anthropic_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Security - REQUIRED: Must be set in environment for production
    jwt_secret_key: str = "dev-secret-key-change-in-production"  # Change this in production!
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    token_refresh_days: int = 7
    
    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    # Database Configuration (Read from environment, not hardcoded)
    # For SQLite: DATABASE_URL=sqlite:///./data/joule_dev.db
    # For PostgreSQL: DATABASE_URL=postgresql://user:password@host:port/db
    database_url: Optional[str] = None
    
    # Individual PostgreSQL settings (for manual construction if needed)
    db_host: Optional[str] = None
    db_port: int = 5432
    db_user: Optional[str] = None
    db_password: Optional[str] = None  # Never hardcode - read from env only
    db_name: str = "joule_replacement"
    
    @property
    def database_connection_string(self) -> str:
        """Generate database connection string from environment variables"""
        if self.database_url:
            return self.database_url
        # Construct from individual params if provided
        if self.db_host and self.db_user and self.db_password:
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        # Default to SQLite for development
        return "sqlite:///./data/joule_dev.db"
    
    # HANA Cloud (Optional)
    hana_host: Optional[str] = None
    hana_port: int = 443
    hana_user: Optional[str] = None
    hana_password: Optional[str] = None
    hana_database: Optional[str] = None
    
    # Vector Store Configuration
    vector_store_type: str = "chroma"  # Options: "chroma", "faiss", "pinecone"
    vector_store_path: str = "./data/vector_store"
    vector_store_collection: str = "joule_knowledge"
    embedding_model: str = "text-embedding-3-small"
    
    # Knowledge Base
    knowledge_base_path: str = "./data/knowledge_base"
    knowledge_base_chunk_size: int = 1000
    knowledge_base_chunk_overlap: int = 200
    
    # Cache Configuration (optional)
    cache_type: str = "memory"  # Options: "memory", "redis"
    cache_ttl_seconds: int = 3600
    redis_url: Optional[str] = None
    
    # Agent Configuration
    max_agent_iterations: int = 10
    agent_timeout_seconds: int = 30
    
    # Feature Flags
    enable_conversation_memory: bool = True
    enable_multi_agent_orchestration: bool = True
    enable_websocket_support: bool = True
    enable_audit_logging: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()