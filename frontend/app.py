import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st
from dotenv import load_dotenv

# Load repo-root .env (same pattern as backend) so BACKEND_URL works locally
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# "backend" only resolves inside Docker Compose. Local dev → use 127.0.0.1
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/v1")


def get_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=BACKEND_URL, timeout=60.0)


async def fetch_sessions() -> List[Dict[str, Any]]:
    async with get_client() as client:
        resp = await client.get("/sessions")
        resp.raise_for_status()
        return resp.json()


async def fetch_history(session_id: str) -> List[Dict[str, Any]]:
    async with get_client() as client:
        resp = await client.get(f"/history/{session_id}")
        resp.raise_for_status()
        return resp.json()


async def send_chat(session_id: Optional[str], question: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    async with get_client() as client:
        resp = await client.post("/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


async def upload_pdfs(files) -> Dict[str, Any]:
    async with get_client() as client:
        f = [("files", (file.name, file.getvalue(), "application/pdf")) for file in files]
        resp = await client.post("/upload-pdf", files=f)
        resp.raise_for_status()
        return resp.json()


def main():
    st.set_page_config(page_title="Advanced RAG Chatbot", layout="wide")
    st.title("Advanced RAG Chatbot")

    # Ensure state keys exist even if Streamlit reruns mid-interaction.
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("messages", [])

    col_sidebar, col_chat = st.columns([1, 3])

    with col_sidebar:
        st.header("Sessions")
        if st.button("New session"):
            st.session_state["session_id"] = None
            st.session_state["messages"] = []

        # Fetch sessions from backend
        try:
            sessions = asyncio.run(fetch_sessions())
        except Exception as e:
            st.error(f"Failed to load sessions: {e}")
            sessions = []

        for s in sessions:
            label = s.get("title") or s.get("id", "")[:8]
            if st.button(label, key=f"session-{s['id']}"):
                st.session_state["session_id"] = s["id"]
                try:
                    history = asyncio.run(fetch_history(s["id"]))
                    st.session_state["messages"] = history
                except Exception as e:
                    st.error(f"Failed to load history: {e}")

        st.header("Upload PDFs")
        files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if files and st.button("Ingest PDFs"):
            try:
                result = asyncio.run(upload_pdfs(files))
                st.success(f"Ingested documents: {len(result.get('document_ids', []))}")
            except Exception as e:
                st.error(f"Failed to ingest PDFs: {e}")

    with col_chat:
        st.header("Chat")
        session_id = st.session_state.get("session_id")
        messages = st.session_state.get("messages", [])
        if session_id:
            st.caption(f"Session: {session_id}")

        history_container = st.container()
        with history_container:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    st.markdown(f"**Assistant:** {content}")

        question = st.text_input("Your question", key="question_input")
        if st.button("Send") and question.strip():
            try:
                resp = asyncio.run(send_chat(session_id, question.strip()))
                st.session_state["session_id"] = resp["session_id"]
                st.session_state["messages"].append(
                    {"role": "user", "content": question.strip()}
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": resp["answer"]}
                )

                st.subheader("Answer")
                st.write(resp["answer"])

                st.subheader("Sources")
                for src in resp.get("sources", []):
                    meta = src.get("metadata", {}) or {}
                    file_name = meta.get("file_name", "unknown")
                    page = meta.get("page_number", "?")
                    st.markdown(f"- **{file_name} (page {page})**")

                st.subheader("Performance Metrics (ms)")
                metrics = resp.get("metrics", {})
                st.json(metrics)
            except Exception as e:
                st.error(f"Failed to send chat: {e}")


if __name__ == "__main__":
    main()

