import loguru
from pathlib import Path
from ProjecX.Llama_index.model_loader import ModelLoader
from ProjecX.Llama_index.data_retirval import DataRetrieval
from ProjecX.Llama_index.Data_ingestion import chunking, VectorStoreManager
from ProjecX.Llama_index.Data_ingestion import Docloader  

class Rag_pipeline:
    def __init__(self):
        self.logger = loguru.logger

        self.model_loader = ModelLoader()
        self.data_retrieval = DataRetrieval()
        self.docloader = Docloader()
        self.chunker = chunking()
        self.vector_store_manager = VectorStoreManager()

   
    def run_pipeline(self, file_path: str, query: str):
        try:
            self.model_loader.load_models()
            self.model_loader.set_settings()

            documents = None
            if not Path(self.vector_store_manager.vector_store_path, "docstore.json").exists():
                text = self.docloader.load_pdf(file_path)
                chunks = self.chunker.chunk_text(text)
                documents = self.chunker.convert_chunks(chunks)

            index = self.vector_store_manager.get_or_create(documents)

            query_engine = index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact"
            )

            response = query_engine.query(query)

            nodes = response.source_nodes or []

            if not nodes:
                return {
                    "answer": response.response,
                    "nodes": [],
                    "top_score": 0.0,
                }

            return {
                "answer": response.response,
                "nodes": nodes,
                "top_score": float(nodes[0].score),
            }

        except Exception as e:
            self.logger.exception("RAG pipeline failed")
            return {
                "answer": "",
                "nodes": [],
            "top_score": 0.0,
        }
                 


