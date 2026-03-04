from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
import loguru
from llama_index.core import Settings


class ModelLoader:
    def __init__(
        self,
        model_name: str = "qwen3.5:397b-cloud",
        embedding_model_name: str = "qwen3-embedding:4b",
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.logger = loguru.logger
        self.llm = None
        self.embed_model = None

    def load_models(self):
        try:
            self.logger.info(f"Loading LLM: {self.model_name}")
            self.llm = Ollama(model=self.model_name)
            
            

            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embed_model = OllamaEmbedding(
                model_name=self.embedding_model_name
            )
           

            self.logger.info("Models loaded successfully")
        except Exception:
            self.logger.exception("Failed to load models")
            raise

    def set_settings(self):
        if self.llm is None or self.embed_model is None:
            raise RuntimeError("Models must be loaded before setting Settings")

        self.logger.info("Setting LlamaIndex global settings")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model