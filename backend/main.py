import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import init_db, close_db
from app.api.routes import api_router
from app.embedding.embedder import get_local_embedder


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Database initialized")

    # Pre-load local embedding model once on startup to avoid long first-request latency.
    # This is especially important on Windows where downloads/model loading can be slow.
    try:
        if (settings.embeddings_provider or "").lower() == "local":
            logger.info("Pre-loading local embedding model...")
            get_local_embedder()
            logger.info("Local embedding model ready.")
    except Exception:
        logger.exception("Local embedding model pre-load failed; continuing anyway.")

    try:
        yield
    finally:
        await close_db()
        logger.info("Database connection closed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Advanced RAG Chatbot Backend",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

