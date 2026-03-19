import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.core.config import settings


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:"
        f"{settings.postgres_password}@{settings.postgres_host}:"
        f"{settings.postgres_port}/{settings.postgres_db}"
    )


async def init_db() -> None:
    global engine, async_session_factory

    if engine is not None:
        return

    url = get_database_url()
    engine = create_async_engine(url, echo=settings.debug, future=True)
    async_session_factory = async_sessionmaker(
        engine, expire_on_commit=False, autoflush=False, autocommit=False
    )

    async with engine.begin() as conn:
        # Import models here to avoid circular import at module level
        from app import models  # noqa: F401

        # Ensure pgvector extension and create tables
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    global engine
    if engine is not None:
        await engine.dispose()
        engine = None


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if async_session_factory is None:
        raise RuntimeError("Database not initialized")
    async with async_session_factory() as session:
        yield session

