from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
from pathlib import Path
import logging
import shutil

from ProjecX.Llama_index.Rag_pipeline import Rag_pipeline
from ProjecX.Auto.team import CustomTeam
from ProjecX.Auto.web_search import web_search_agent
from evaluate import ask_with_full_evaluation

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- App ----------------
app = FastAPI(title="ProjectX RAG Server", version="2.0.0")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"


UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "ProjectX Hybrid RAG server running"}

# ---------------- Core ----------------
rag_pipeline = Rag_pipeline()

team = CustomTeam(
    rag_pipeline=rag_pipeline,
    web_agent=web_search_agent(),
)

# ---------------- Schemas ----------------
class UploadResponse(BaseModel):
    doc_id: str
    message: str


class QueryRequest(BaseModel):
    doc_id: str
    query: str
    evaluate:bool = False


class QueryResponse(BaseModel):
    source: str
    answer: str
    score: float
    evaluation:dict|None = None


# ---------------- Upload (INGEST) ----------------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}.pdf"

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # 🔥 IMPORTANT: await
        await rag_pipeline.ingest(
            file_path=str(file_path),
            persist_dir=doc_id,   # now used as collection_name
        )

        logging.info(f"Ingested doc_id={doc_id}")

    except Exception as e:
        logging.error(f"Ingestion failed: {e}")

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(500, "Failed to ingest document")

    return UploadResponse(
        doc_id=doc_id,
        message="File uploaded and indexed successfully",
    )


# ---------------- Query ----------------
@app.post("/query", response_model=QueryResponse)
async def query_doc(req: QueryRequest):

    try:
        # ---- NORMAL FLOW ----
        if not req.evaluate:
            result = await team.run(
                query=req.query,
                doc_id=req.doc_id,
            )

            logging.info(f"Query routed to {result['source']}")

            return QueryResponse(
                source=result.get("source", ""),
                answer=result.get("answer", ""),
                score=result.get("score", 0),
                evaluation=None
            )

        # ---- EVALUATION FLOW (uses your function) ----
        eval_result = await ask_with_full_evaluation(
            query=req.query,
            pipeline=rag_pipeline,
            persist_dir=req.doc_id 
        )

        return QueryResponse(
            source="rag_pipeline_eval",
            answer=eval_result["answer"],
            score=0.0,  
            evaluation=eval_result["evaluation"]
        )

    except Exception as e:
        logging.error(f"Query failed: {e}")
        raise HTTPException(500, "Query failed")