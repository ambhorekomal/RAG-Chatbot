import os
from functools import lru_cache

from pydantic import AnyHttpUrl, BaseModel, Field
from dotenv import load_dotenv


# Load ../.env once at import time
# repo root = .../rag chatbot/
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
# Force .env to apply even if the variables exist but are empty.
# This prevents "GEMINI_API_KEY missing" when an empty env var is present.
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


class Settings(BaseModel):
    # App
    app_name: str = "Advanced RAG Chatbot"
    environment: str = Field(default=os.getenv("ENVIRONMENT", "development"))
    debug: bool = Field(default=os.getenv("DEBUG", "true").lower() == "true")

    # Database (simple default: postgres/postgres on default DB)
    postgres_host: str = Field(default=os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = Field(default=int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = Field(default=os.getenv("POSTGRES_DB", "postgres"))
    postgres_user: str = Field(default=os.getenv("POSTGRES_USER", "postgres"))
    postgres_password: str = Field(default=os.getenv("POSTGRES_PASSWORD", "postgres"))

    # Redis
    redis_host: str = Field(default=os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = Field(default=int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = Field(default=int(os.getenv("REDIS_DB", "0")))

    # Gemini / LLM
    gemini_api_key: str = Field(default=os.getenv("GEMINI_API_KEY", ""))
    gemini_embedding_model: str = Field(
        default=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    )
    gemini_chat_model: str = Field(
        default=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    )

    # Control Gemini usage to reduce token/quota consumption.
    # Default to retrieval-only rewriting/variants (cheaper) but keep answer on.
    gemini_query_rewrite_enabled: bool = Field(
        default=os.getenv("GEMINI_QUERY_REWRITE_ENABLED", "false").lower() == "true"
    )
    gemini_query_variants_enabled: bool = Field(
        default=os.getenv("GEMINI_QUERY_VARIANTS_ENABLED", "false").lower() == "true"
    )
    gemini_answer_enabled: bool = Field(
        default=os.getenv("GEMINI_ANSWER_ENABLED", "true").lower() == "true"
    )

    # BGE / Reranker
    bge_embedding_model_name: str = Field(
        default=os.getenv("BGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    )
    bge_reranker_model_name: str = Field(
        default=os.getenv("BGE_RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
    )

    # Embeddings provider:
    # - "gemini": use GEMINI embedding model (paid; quota-limited)
    # - "local": use open-source BGE embeddings via sentence-transformers (free)
    embeddings_provider: str = Field(
        default=os.getenv("EMBEDDINGS_PROVIDER", "local")
    )
    local_embeddings_batch_size: int = Field(
        default=int(os.getenv("LOCAL_EMBEDDINGS_BATCH_SIZE", "32"))
    )
    local_embeddings_device: str = Field(
        default=os.getenv("LOCAL_EMBEDDINGS_DEVICE", "cpu")
    )

    # pgvector column dimensions in your current database schema.
    # Your current DB expects 1536-dim text vectors (documents.embedding),
    # while bge-large-en-v1.5 is 1024-dim. We pad/truncate embeddings to match.
    pgvector_text_embedding_dim: int = Field(
        default=int(os.getenv("PGVECTOR_TEXT_EMBEDDING_DIM", "1536"))
    )
    pgvector_image_embedding_dim: int = Field(
        default=int(os.getenv("PGVECTOR_IMAGE_EMBEDDING_DIM", "768"))
    )

    # Performance knobs
    reranker_enabled: bool = Field(
        default=os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    )
    retrieval_top_k: int = Field(
        default=int(os.getenv("RETRIEVAL_TOP_K", "10"))
    )
    reranker_top_k: int = Field(
        default=int(os.getenv("RERANKER_TOP_K", "5"))
    )

    # PDF ingestion speed controls
    extract_images: bool = Field(
        default=os.getenv("EXTRACT_IMAGES", "false").lower() == "true"
    )
    ingest_max_pages: int = Field(
        default=int(os.getenv("INGEST_MAX_PAGES", "200000"))
    )

    # Retrieval speed knobs
    bm25_enabled: bool = Field(
        default=os.getenv("BM25_ENABLED", "false").lower() == "true"
    )

    # Cohere (optional reranker)
    cohere_api_key: str = Field(default=os.getenv("COHERE_API_KEY", ""))

    # PDF / ingestion
    max_upload_size_mb: int = Field(
        default=int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    )

    # CORS
    backend_cors_origins: list[AnyHttpUrl] = []


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

