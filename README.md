# Projectx

Projectx is a FastAPI-based backend that combines **LlamaIndex** (for Retrieval-Augmented Generation) and **AutoGen-style agents** (for task orchestration).

It is designed to answer queries using indexed documents, optional web context, and agent-driven control flow.

---

## Tech Stack

- Python 3.12
- FastAPI
- LlamaIndex (RAG)
- AutoGen (agent orchestration)
- Docker

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn ProjecX.main:app --host 0.0.0.0 --port 8000