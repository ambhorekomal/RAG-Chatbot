import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db_session
from app.embedding.embedder import embed_and_store_documents
from app.models.document import Document
from app.services.ingestion.pdf_ingestion import ingest_pdf


router = APIRouter()


UPLOAD_DIR = Path(os.getenv("PDF_UPLOAD_DIR", "storage/pdfs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload-pdf")
async def upload_pdf(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db_session),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    created_ids: List[uuid.UUID] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        page_doc_ids = await ingest_pdf(db, f, storage_dir=str(UPLOAD_DIR))
        created_ids.extend(page_doc_ids)

    # Embed newly created documents
    docs_result = await db.execute(
        select(Document).where(Document.id.in_(created_ids))
    )
    docs = docs_result.scalars().all()
    if docs:
        await embed_and_store_documents(db, docs)

    await db.commit()

    return {"document_ids": [str(i) for i in created_ids]}

