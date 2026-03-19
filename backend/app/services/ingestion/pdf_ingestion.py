import io
import logging
import uuid
from typing import Any, List

import fitz  # PyMuPDF
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentImage
from app.core.config import settings


logger = logging.getLogger(__name__)


async def ingest_pdf(
    db: AsyncSession,
    file: UploadFile,
    storage_dir: str,
) -> List[uuid.UUID]:
    """
    Ingest a PDF: extract text, images, and create document records.
    Returns list of created document IDs (per chunk).
    """
    raw_bytes = await file.read()
    doc = fitz.open(stream=io.BytesIO(raw_bytes), filetype="pdf")

    documents: list[Document] = []
    images: list[DocumentImage] = []

    max_pages = int(settings.ingest_max_pages)
    for page_index in range(min(len(doc), max_pages)):
        page = doc[page_index]
        page_number = page_index + 1

        text = page.get_text("text")
        blocks = page.get_text("blocks")

        meta: dict[str, Any] = {
            "file_name": file.filename,
            "page_number": page_number,
        }

        # For now, store per-page; later the chunker will split into token-sized chunks.
        document = Document(
            content=text,
            extra_metadata=meta,
        )
        documents.append(document)

        if settings.extract_images:
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                img_uuid = uuid.uuid4()
                image_path = f"{storage_dir}/{img_uuid}.png"
                if pix.n - pix.alpha < 4:
                    pix.save(image_path)
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(image_path)
                pix = None

                image_meta = {
                    "file_name": file.filename,
                    "page_number": page_number,
                    "image_index": img_index,
                }
                images.append(
                    DocumentImage(
                        document=document,
                        page_number=page_number,
                        image_path=image_path,
                        extra_metadata=image_meta,
                    )
                )

    db.add_all(documents)
    if images:
        db.add_all(images)
    await db.flush()

    doc_ids = [d.id for d in documents]
    logger.info("Ingested %d pages from %s", len(documents), file.filename)

    return doc_ids

