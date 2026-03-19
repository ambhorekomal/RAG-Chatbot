from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.rag_pipeline import run_rag_pipeline
from app.db.database import get_db_session
from app.models.chat import ChatMessage, ChatSession


router = APIRouter()


class ChatRequest(BaseModel):
    session_id: Optional[uuid.UUID] = None
    question: str


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, float]


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session: ChatSession
    if payload.session_id:
        session = await db.get(ChatSession, payload.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = ChatSession()
        db.add(session)
        await db.flush()

    user_msg = ChatMessage(session_id=session.id, role="user", content=payload.question)
    db.add(user_msg)
    await db.flush()

    rag_result = await run_rag_pipeline(db, payload.question)

    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=rag_result["answer"],
    )
    db.add(assistant_msg)
    await db.commit()

    return ChatResponse(
        session_id=session.id,
        answer=rag_result["answer"],
        sources=rag_result["sources"],
        metrics=rag_result["metrics"],
    )

