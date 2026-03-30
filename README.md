# 🚀 ProjectX – Hybrid Multi-Agent RAG System

ProjectX is a modular AI system that combines **Retrieval-Augmented Generation (RAG)**, **web search**, and a **reranking layer** using a **multi-agent architecture built on FastAPI** to generate complete, accurate, and up-to-date responses.

---

## 🔑 Problem It Solves

Traditional RAG systems rely only on ingested documents:

* Missing information → system fails
* Partial information → incomplete answers

---

## ✅ Solution (ProjectX)

ProjectX introduces a **hybrid knowledge system**:

* 📚 Uses **RAG** for document-based knowledge
* 🌐 Uses **web search** for missing information
* 🔀 Combines both when knowledge is partial
* 🧠 Uses a **reranker (BAAI/bge-reranker-v2-m3)** to select the most relevant context

---

## 🎯 Result

* More complete answers
* Better accuracy
* Improved relevance via reranking
* Real-time information support

---

## ⚙️ Key Features

* Multi-agent architecture (routing + coordination)
* Retrieval-Augmented Generation (RAG)
* Web search integration
* Reranking layer (cross-encoder)
* FastAPI backend
* Vector database (ChromaDB)
* Modular system design

---

## 🏗️ Architecture

User Query
↓
FastAPI Endpoint
↓
Supervisor / Router
↓
-

## | RAG Agent | Web Agent | Hybrid |

↓
Reranker (BAAI/bge-reranker-v2-m3)
↓
LLM Response Generator
↓
Final Output

---

## 🧠 How It Works

1. Request comes through FastAPI
2. Supervisor agent analyzes query intent
3. Routes query:

   * RAG → stored knowledge
   * Web → external info
   * Hybrid → both
4. Retrieved chunks are reranked
5. Top context is selected
6. LLM generates final response

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

ProjectX improves retrieval quality using a cross-encoder reranker:

```python id="rerank01"
from llama_index.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=5
)
```

Only the most relevant chunks are passed to the LLM, improving accuracy and reducing noise.

---

## 🛠️ Tech Stack

* Python
* FastAPI
* LlamaIndex
* ChromaDB
* SentenceTransformers (bge-reranker-v2-m3)
* Custom Multi-Agent System

---

## 📂 Project Structure

Projectx/
│
├── main.py
├── requirements.txt
├── Dockerfile
│
├── ProjecX/
│   ├── Auto/              # Agents
│   │   ├── Rag_agent.py
│   │   ├── team.py
│   │   ├── web_search.py
│   │   ├── prompt.py
│   │   └── model.py
│   │
│   ├── Llama_index/       # RAG pipeline
│       ├── Rag_pipeline.py
│       ├── data_retrieval.py
│       ├── chroma_client.py
│       └── model_loader.py

---

## ⚡ Run Locally

```bash id="runlocal01"
git clone https://github.com/tanishq450/Projectx.git
cd Projectx
pip install -r requirements.txt
python main.py
```

---

## 🐳 Run with Docker

### Build Image

```bash id="dockerbuild01"
docker build -t projectx .
```

### Run Container

```bash id="dockerrun01"
docker run -it --env-file .env -p 8000:8000 projectx
```

---

## 👨‍💻 Author

Tanishq Kumar
