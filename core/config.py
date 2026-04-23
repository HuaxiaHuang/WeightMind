import os

# 🚀 核心新增：强制 Python 在访问本地服务时绕过 Clash 等代理！
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1,dashscope.aliyuncs.com"

# 🚀 2. 核心新增：强制开启离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────
    LLM_PROVIDER: str = "openai"
    LLM_API_KEY: str = "sk-placeholder"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_SUMMARY_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1

    # ── Embedding & Reranker ─────────────────────────────
    EMBEDDING_MODEL_PATH: str = "E:/WorkBuddy/Sci_RAG/models/bge-m3"
    RERANKER_MODEL_PATH: str = "E:/WorkBuddy/Sci_RAG/models/bge-reranker-v2-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    # ── PostgreSQL ───────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://sci_rag:sci_rag_pass@localhost:5432/sci_rag_db"
    )

    # ── Qdrant ───────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # ── Redis / Celery ───────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── 🚀 新增：云端解析引擎 (TextIn & LlamaParse) ────────
    TEXTIN_APP_ID: str = ""
    TEXTIN_SECRET_CODE: str = ""
    LLAMA_CLOUD_API_KEY: str = ""
    PARSE_MODE: str = "eco" # 支持 "eco" (节能轮询) 或 "performance" (性能并发)

    # ── 文件存储 ──────────────────────────────────────────
    DATA_ROOT: Path = Path("E:/WorkBuddy/Sci_RAG/data")
    RAW_DATA_DIR: Path = Path("E:/WorkBuddy/Sci_RAG/data/raw")
    TEMP_DIR: Path = Path("E:/WorkBuddy/Sci_RAG/data/temp")
    MAX_UPLOAD_SIZE_MB: int = 100

    # ── RAG 参数 ──────────────────────────────────────────
    CHILD_CHUNK_SIZE: int = 128
    PARENT_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 20
    TOP_K_RETRIEVE: int = 10
    TOP_K_RERANK: int = 3
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3

    # ── 应用 ─────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()