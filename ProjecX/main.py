from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
import os

from Llama_index.Rag_pipeline import Rag_pipeline
from Auto.team import CustomTeam
from Auto.web_search import web_search_agent

app = FastAPI(title="Document RAG Server")

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vector_stores"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

rag_pipeline = Rag_pipeline()
team = CustomTeam(rag_pipeline, web_search_agent())


# ---------- Schemas ----------
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


# ---------- Upload ----------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    doc_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "doc_id": doc_id,
        "message": "File uploaded successfully. Use doc_id for queries.",
    }


# ---------- Query ----------
@app.post("/query", response_model=QueryResponse)
async def query_doc(req: QueryRequest):
    file_path = os.path.join(UPLOAD_DIR, f"{req.doc_id}.pdf")
    persist_dir = os.path.join(VECTOR_DIR, req.doc_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")

    result = await team.run(
        query=req.query,
        file_path=file_path,
        persist_dir=persist_dir,
    )

    return {
        "source": result["source"],
        "answer": result["answer"],
        "score": float(result["score"]),
    }