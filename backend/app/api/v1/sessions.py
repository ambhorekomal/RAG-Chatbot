from typing import List
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db_session
from app.models.chat import ChatMessage, ChatSession


router = APIRouter()


class SessionSummary(BaseModel):
    id: uuid.UUID
    title: str | None


class ChatMessageOut(BaseModel):
    role: str
    content: str


@router.get("/history/{session_id}", response_model=List[ChatMessageOut])
async def get_history(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [ChatMessageOut(role=m.role, content=m.content) for m in rows]


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()
    return {"status": "deleted"}


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(
    db: AsyncSession = Depends(get_db_session),
):
    rows = (await db.execute(select(ChatSession))).scalars().all()
    return [SessionSummary(id=s.id, title=s.title) for s in rows]

