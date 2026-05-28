from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Production RAG AI Assistant"
    DEBUG: bool = True

    # LLM
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-3.5-turbo"
    USE_LOCAL_MODEL: bool = False
    OLLAMA_BASE_URL: Optional[str] = None        # e.g. http://localhost:11434
    OLLAMA_MODEL: str = "llama3"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # local HuggingFace model

    # Vector DB
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # Memory
    MEMORY_DB_URL: str = "sqlite:///./memory.db"

    # API
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()