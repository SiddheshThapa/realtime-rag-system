# services/utils/settings.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    kafka_bootstrap: str = "kafka:9092"
    postgres_host: str = "postgres"
    postgres_db: str = "ragdb"
    postgres_user: str = "rag"
    postgres_password: str = "ragpwd"
    redis_host: str = "redis"
    redis_port: int = 6379
    jwt_secret: str = "changeme"
    env: str = "dev"
    ollama_url: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
