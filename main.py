from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
import os
from pathlib import Path

from ProjecX.Llama_index.Rag_pipeline import Rag_pipeline
from ProjecX.Auto.team import CustomTeam
from ProjecX.Auto.web_search import web_search_agent

# ---------------- App ----------------
app = FastAPI(title="Document RAG Server", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "vector_stores"

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

rag_pipeline = Rag_pipeline()


# ---------------- Core objects ----------------
rag_pipeline = Rag_pipeline()
team = CustomTeam(
    rag_pipeline=rag_pipeline,
    web_agent=web_search_agent(),
    vector_dir=VECTOR_DIR,
)

# ---------------- Schemas ----------------
class UploadResponse(BaseModel):
    doc_id: str
    message: str


class QueryRequest(BaseModel):
    doc_id: str
    query: str


class QueryResponse(BaseModel):
    source: str
    answer: str
    score: float


# ---------------- Upload (INGESTION ONLY) ----------------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}.pdf"
    persist_dir = VECTOR_DIR / doc_id

    # Save PDF
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Ingest document
    try:
        rag_pipeline.ingest(
            file_path=str(file_path),
            persist_dir=str(persist_dir),
        )
    except Exception:
        # Cleanup on failure
        if file_path.exists():
            file_path.unlink()
        if persist_dir.exists():
            for p in persist_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            persist_dir.rmdir()

        raise HTTPException(
            status_code=500,
            detail="Failed to ingest document",
        )

    return UploadResponse(
        doc_id=doc_id,
        message="File uploaded and indexed successfully",
    )


# ---------------- Query (DECISION LAYER) ----------------
@app.post("/query", response_model=QueryResponse)
async def query_doc(req: QueryRequest):
    persist_dir = VECTOR_DIR / req.doc_id

    if not persist_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Vector index not found for this document",
        )

    result = await team.run(
        query=req.query,
        doc_id=req.doc_id,
    )

    return QueryResponse(
        source=result["source"],
        answer=result["answer"],
        score=result["score"],
    )