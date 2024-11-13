from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    PROJECT_NAME: str = "ArXiv Paper Analyzer"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API Keys
    OPENAI_API_KEY: str | None = None

    # ArXiv Settings
    ARXIV_BASE_URL: str = "http://export.arxiv.org/api/query"
    DEFAULT_SEARCH_QUERY: str = "RAG LLM"
    MAX_RESULTS: int = 1

    # LLM Settings
    DEFAULT_MODEL: str = "gpt-4-turbo-preview"
    FALLBACK_MODEL: str = "llama3.2"

    # Content Processing
    CHUNK_SIZE: int = 20000

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()