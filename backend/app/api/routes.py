from fastapi import APIRouter


from app.api.v1 import chat, ingestion, sessions


api_router = APIRouter()

api_router.include_router(ingestion.router, prefix="/v1", tags=["ingestion"])
api_router.include_router(chat.router, prefix="/v1", tags=["chat"])
api_router.include_router(sessions.router, prefix="/v1", tags=["sessions"])

