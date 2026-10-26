from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "SynapseGateway"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/v1"
    
    # Gateway Security
    GATEWAY_API_KEY: Optional[str] = None  # If set, clients must pass Bearer token
    
    # Semantic Cache Settings
    CACHE_ENABLED: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = 0.90  # Cosine similarity (0.0 to 1.0)
    CACHE_MAX_ENTRIES: int = 10000
    CACHE_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    USE_LOCAL_ONNX: bool = False  # Set to True in production to download & use full ONNX model
    
    # Guardrails
    GUARDRAILS_ENABLED: bool = True
    BLOCK_PROMPT_INJECTION: bool = True
    MASK_PII: bool = True
    
    # Provider Settings (API Keys will be read from environment)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Circuit Breaker & Retry Settings
    MAX_PROVIDER_RETRIES: int = 2
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
    CIRCUIT_BREAKER_RESET_TIMEOUT_SEC: float = 30.0
    REQUEST_TIMEOUT_SEC: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()
