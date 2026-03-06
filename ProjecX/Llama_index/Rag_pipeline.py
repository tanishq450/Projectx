import loguru
from pathlib import Path
from ProjecX.Llama_index.model_loader import ModelLoader
from ProjecX.Llama_index.data_retirval import DataRetrieval
from ProjecX.Llama_index.Data_ingestion import chunking, VectorStoreManager
from ProjecX.Llama_index.Data_ingestion import Docloader  
from llama_index.core.postprocessor import SentenceTransformerRerank



reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-large",
    top_n=5
)


class Rag_pipeline:
    def __init__(self):
        self.logger = loguru.logger

        self.model_loader = ModelLoader()
        self.docloader = Docloader()
        self.chunker = chunking()
        self.vector_store_manager = VectorStoreManager()

    # ---------------- INGESTION ----------------
    def ingest(self, file_path: str, persist_dir: str) -> None:
        """
        One-time operation.
        Builds and persists a vector index for a document.
        """

        try:
            # Load models once per process if possible
            self.model_loader.load_models()
            self.model_loader.set_settings()

            # Load and preprocess document
            text = self.docloader.load_pdf(file_path)
            chunks = self.chunker.chunk_text(text)
            documents = self.chunker.convert_chunks(chunks)

            if not documents:
                raise RuntimeError("No documents produced from PDF")

            # Create and persist vector store
            self.vector_store_manager.create(
                documents=documents,
                persist_dir=persist_dir,
            )

            self.logger.info(
                f"Document indexed successfully at {persist_dir}"
            )

        except Exception:
            self.logger.exception("RAG ingestion failed")
            raise

    # ---------------- QUERY ----------------
    def query(self, query: str, persist_dir: str) -> dict:
        try:
            self.model_loader.load_models()
            self.model_loader.set_settings()

            index = self.vector_store_manager.load(persist_dir)

            query_engine = index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact",
                verbose=True,
                node_postprocessors=[reranker]
            )

            response = query_engine.query(query)
            nodes = response.source_nodes or []

            score = float(nodes[0].score) if nodes else 0.0

            return {
                "answer": response.response,
                "score": score,
                "nodes": nodes,
            }

        except Exception:
            self.logger.exception("RAG query failed")
            raise