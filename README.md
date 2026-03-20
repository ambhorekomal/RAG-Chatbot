# Advanced RAG Chatbot

This project is a production-ready Retrieval-Augmented Generation (RAG) chatbot for multi-PDF research documents, with:

- FastAPI backend (async)
- PostgreSQL + pgvector for document storage and vector search
- Redis caching for embeddings and retrieval results
- Advanced RAG pipeline (query rewrite, multi-query, hybrid retrieval, optional reranking, context compression)
- Chat history in PostgreSQL
- Streamlit frontend
- Dockerized stack via `docker-compose`

## Quickstart (Docker)

1. Create/edit the root `.env` and set at least:

   - `GEMINI_API_KEY` (required if `GEMINI_ANSWER_ENABLED=true`)

2. Example `.env` (place this at the project root):

   ```env
   # Frontend (Streamlit) talks to backend at this URL (local dev)
   BACKEND_URL=http://127.0.0.1:8000/api/v1

   # Postgres + pgvector (only used for local dev; Docker Compose uses service names)
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=rag_db
   POSTGRES_USER=postgres_user_you_created
   POSTGRES_PASSWORD=postgres_password_of_your_db

   # Redis (optional; if not running, caching is disabled)
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0

   # Gemini (used for query rewrite/variants and/or answer generation if enabled)
   GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
   GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
   GEMINI_CHAT_MODEL=gemini-2.5-flash

   GEMINI_QUERY_REWRITE_ENABLED=false
   GEMINI_QUERY_VARIANTS_ENABLED=false
   GEMINI_ANSWER_ENABLED=true

   # Embeddings provider: "local" (free) or "gemini" (paid)
   EMBEDDINGS_PROVIDER=local
   LOCAL_EMBEDDINGS_BATCH_SIZE=64
   LOCAL_EMBEDDINGS_DEVICE=cpu

   # Speed/cost knobs
   RERANKER_ENABLED=false
   RETRIEVAL_TOP_K=6
   RERANKER_TOP_K=5
   BM25_ENABLED=false

   # PDF ingestion speed
   EXTRACT_IMAGES=false
   INGEST_MAX_PAGES=60

   MAX_UPLOAD_SIZE_MB=50
   ```

3. Build and start the stack:

   - `docker-compose up --build`

4. Services:

   - Backend API: `http://localhost:8000/api`
   - Streamlit frontend: `http://localhost:8501`

## Backend API

Base URL (inside Docker): `http://backend:8000/api/v1`  
Base URL (locally): `http://localhost:8000/api/v1`

### `POST /upload-pdf`

Upload one or more PDFs for ingestion.

- **Body**: multipart/form-data with `files` (one or more PDF files).
- **Response**:

  ```json
  { "document_ids": ["..."] }
  ```

### `POST /chat`

Ask a question against all ingested documents.

- **Body (JSON)**:

  ```json
  {
    "session_id": null,
    "question": "What is the main contribution of the paper?"
  }
  ```

- **Response**:

  ```json
  {
    "session_id": "uuid",
    "answer": "…",
    "sources": [
      {
        "id": "uuid",
        "content": "…",
        "metadata": { "file_name": "paper.pdf", "page_number": 3 },
        "score": 0.9,
        "rerank_score": 12.3
      }
    ],
    "metrics": {
      "query_rewrite_ms": 10.0,
      "retrieval_total_ms": 50.0,
      "vector_search_ms": 20.0,
      "bm25_search_ms": 15.0,
      "merge_ms": 5.0,
      "embedding_ms": 30.0,
      "rerank_ms": 25.0,
      "llm_ms": 200.0,
      "context_chars": 3000
    }
  }
  ```

### `GET /history/{session_id}`

Return the chat history (messages) for a given session.

### `DELETE /session/{session_id}`

Delete a session and its messages.

### `GET /sessions`

List all chat sessions.

## Local Development (no Docker)

1. Create a virtual environment and install dependencies:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Ensure PostgreSQL (with `pgvector`) is running.

   - If you use your existing pgvector Docker container, make sure `.env` matches it, e.g.:
     - `POSTGRES_HOST=localhost`
     - `POSTGRES_DB=rag_db`
     - `POSTGRES_USER=postgres`
     - `POSTGRES_PASSWORD=postgres`

   - Redis is optional (if it isn't running, caching is automatically disabled).

   Example pgvector container run:

   ```bash
   docker run -d --name pgvector-db ^
     -e POSTGRES_USER=postgres ^
     -e POSTGRES_PASSWORD=postgres ^
     -e POSTGRES_DB=rag_db ^
     -p 5432:5432 ^
     ankane/pgvector
   ```

3. Start the backend:

   ```bash
   uvicorn main:app --reload
   ```

4. Start the Streamlit frontend:

   ```bash
   cd ../frontend
   streamlit run app.py
   ```

## Notes

- Image extraction and storage are implemented at ingestion; image embeddings/captions can be extended using CLIP/BLIP in `app/embedding/embedder.py`.
- Reranking uses a BGE cross-encoder model; ensure there is enough memory for it in production.
- If you want faster ingestion and faster answers, use the speed knobs in the root `.env`:
  - `EXTRACT_IMAGES=false`
  - `INGEST_MAX_PAGES` (e.g. 60)
  - `RERANKER_ENABLED=false`
  - `BM25_ENABLED=false`
- For production, configure proper CORS, logging, and secure secrets management instead of `.env`.
