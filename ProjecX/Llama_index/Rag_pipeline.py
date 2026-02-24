import loguru
from pathlib import Path
from Llama_index.model_loader import ModelLoader
from Llama_index.data_retirval import DataRetrieval
from Llama_index.Data_ingestion import chunking,VectorStoreManager  
from Llama_index.Data_ingestion import Docloader  

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
            # 1️⃣ Load models once
            self.model_loader.load_models()
            self.model_loader.set_settings()

            # 2️⃣ Ingest ONLY if vector store does not exist
            documents = None
            if not Path(self.vector_store_manager.vector_store_path, "docstore.json").exists():
                self.logger.info("Vector store not found. Running ingestion.")

                text = self.docloader.load_pdf(file_path)
                chunks = self.chunker.chunk_text(text)
                documents = self.chunker.convert_chunks(chunks)

            # 3️⃣ Get or create index
            index = self.vector_store_manager.get_or_create(documents)

            # 4️⃣ Query
            query_engine = index.as_query_engine()
            response = query_engine.query(query)

            return response.response

        except Exception as e:
            self.logger.exception(f"Error running pipeline: {e}")
            return None




"""if __name__ == "__main__":
    rag_pipeline = Rag_pipeline()
    response = rag_pipeline.run_pipeline(
       "/home/tanishq/ProjecX/Llama_index/1706.03762v7.pdf",
       "explain Transformer architecture"
    )
    print(response)"""