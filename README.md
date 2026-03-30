# 🚀 ProjectX – Hybrid Multi-Agent RAG System (Qdrant + RRF + Reranker)

ProjectX is a modular AI system that combines **Retrieval-Augmented Generation (RAG)**, **web search**, and **hybrid retrieval (dense + sparse)** using a **multi-agent architecture built on FastAPI** to generate accurate, complete, and up-to-date responses.

---

## 🔑 Problem It Solves

Traditional RAG systems rely only on ingested documents:

- ❌ Missing information → system fails  
- ❌ Partial information → incomplete answers  

---

## ✅ Solution (ProjectX)

ProjectX introduces a **hybrid knowledge system**:

- 📚 Uses **RAG** for document-based knowledge  
- 🌐 Uses **web search** for missing information  
- 🔀 Combines both when knowledge is partial  
- 🧠 Uses **RRF (Reciprocal Rank Fusion)** for hybrid retrieval  
- 🎯 Uses **reranker (BAAI/bge-reranker-v2-m3)** for final relevance  

---

## 🎯 Result

- More complete answers  
- Better accuracy  
- Improved relevance via reranking  
- Real-time information support  

---

## ⚙️ Key Features

- Multi-agent architecture (routing + coordination)  
- Hybrid retrieval (Dense + BM25 Sparse)  
- Qdrant vector database  
- RRF fusion inside database  
- Cross-encoder reranker  
- Web search integration  
- FastAPI backend  
- Modular system design  

---

## 🏗️ Architecture
User Query
↓
FastAPI Endpoint
↓
Supervisor / Router
↓
┌───────────────┬───────────────┬───────────────┐
│ RAG Agent │ Web Agent │ Hybrid │
└───────────────┴───────────────┴───────────────┘
↓
Qdrant Hybrid Retrieval (Dense + Sparse + RRF)
↓
Reranker (BAAI/bge-reranker-v2-m3)
↓
Final Answer



---

## 🧠 How It Works

1. Request comes through FastAPI  
2. Supervisor agent analyzes query intent  
3. Routes query:
   - RAG → stored document knowledge  
   - Web → external search  
   - Hybrid → combines both  
4. Qdrant performs hybrid retrieval (dense + sparse + RRF)  
5. Retrieved chunks are reranked  
6. Top context is selected  
7. Final response is generated  

---

## 💡 Example Use Cases

**Query:** What is transformer architecture?  
→ RAG  

**Query:** Latest AI news  
→ Web search  

**Query:** Explain LLMs with latest advancements  
→ Hybrid + Reranker  

---

## 🔬 Reranking

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=5
)


🔍 Retrieval Strategy
Dense → semantic understanding
Sparse → keyword matching
RRF → combines both
Reranker → final refinement


🛠️ Tech Stack
Python
FastAPI
Qdrant
FastEmbed
SentenceTransformers
LlamaIndex
Multi-Agent System



Projectx/
│
├── main.py
├── requirements.txt
├── Dockerfile
│
├── ProjecX/
│   ├── Auto/
│   │   ├── team.py
│   │   ├── web_search.py
│   │
│   ├── Llama_index/
│   │   ├── Rag_pipeline.py
│   │   ├── data_ingestion.py
│   │   ├── sparse.py
│   │   └── model_loader.py
│
├── qdrant.py
├── tests/


🐳 Run Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

▶️ Run Server
```bash
uvicorn main:app --reload
```

🔌 API Usage
Upload
```bash
curl -X POST http://localhost:8000/upload \
-F "file=@sample.pdf"
```

Query
```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{
  "doc_id": "your-doc-id",
  "query": "What is transformer architecture?"
}'


👨‍💻 Author

Tanishq Kumar