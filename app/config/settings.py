"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application
    app_name: str = "Joule Replacement"
    app_version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # LLM Configuration
    openai_api_key: str
    
    # Security
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    # Database (Optional)
    hana_host: str | None = None
    hana_port: int = 443
    hana_user: str | None = None
    hana_password: str | None = None
    hana_database: str | None = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()