import loguru
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)


class DataRetrieval:
    def __init__(self, vector_store_path: str = "./vector_store"):
        self.vector_store_path = vector_store_path
        self.logger = loguru.logger
        self.index = None
        self.query_engine = None

    def load_vector_store(self):
        try:
            self.logger.info(
                f"Loading vector store from {self.vector_store_path}"
            )

            storage_context = StorageContext.from_defaults(
                persist_dir=self.vector_store_path
            )

            self.index = load_index_from_storage(storage_context)
            self.query_engine = self.index.as_query_engine()

            self.logger.info("Vector store loaded successfully")

        except Exception:
            self.logger.exception("Failed to load vector store")
            raise

    def query(self, query: str):
        if self.query_engine is None:
            raise RuntimeError(
                "Vector store not loaded. Call load_vector_store() first."
            )

        self.logger.info(f"Querying vector store: {query}")
        response = self.query_engine.query(query)
        return response