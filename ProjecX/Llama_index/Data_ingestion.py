from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
import fitz
import os
import loguru
from chonkie import RecursiveChunker
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from pathlib import Path
from llama_index.core import load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss


class Docloader:
    def __init__(self,output_dir="./data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir,exist_ok=True)
        self.logger = loguru.logger
        
    def load_pdf(self,file_path:str):
        try:
            self.logger.info(f"Loading PDF from {file_path}")
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            self.logger.info(f"PDF loaded successfully")
            return text
        except Exception as e:
            self.logger.error(f"Error loading PDF: {e}")
            return None 

    def save_text(self,text:str,file_name:str):
        try:
            self.logger.info(f"Saving text to {file_name}")
            with open(os.path.join(self.output_dir,file_name),"w") as f:
                f.write(text)
            self.logger.info(f"Text saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving text: {e}")

    
    def is_encrypted(self,file_path:str):
        try:
            self.logger.info(f"Checking if PDF is encrypted: {file_path}")
            doc = fitz.open(file_path)
            if doc.is_encrypted:
                self.logger.info(f"PDF is encrypted")
                return True
            else:
                self.logger.info(f"PDF is not encrypted")
                return False
        except Exception as e:
            self.logger.error(f"Error checking if PDF is encrypted: {e}")
            return None 
        
        

  

    

class chunking:
    def __init__(self, chunk_size: int = 1000, output_dir: str = "./data"):
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.logger = loguru.logger
        
    def chunk_text(self,text:str):
        try:
            self.logger.info(f"Chunking text")
            chunker = RecursiveChunker(chunk_size=self.chunk_size)
            chunks = chunker.chunk(text)
            self.logger.info(f"Text chunked successfully")
            return chunks
        except Exception as e:
            self.logger.error(f"Error chunking text: {e}")
            return None 


    def save_chunks(self,chunks:list,file_name:str):
        try:
            self.logger.info(f"Saving chunks to {file_name}")
            with open(os.path.join(self.output_dir,file_name),"w") as f:
                for chunk in chunks:
                    f.write(chunk.text + "\n")
            self.logger.info(f"Chunks saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving chunks: {e}")
    

    def convert_chunks(self, chunks: list):
        if not chunks:
            raise ValueError("No chunks provided")

        self.logger.info("Converting chunks to Documents")

        documents = []
        for chunk in chunks:
            if isinstance(chunk, str):
                documents.append(Document(text=chunk))
            elif hasattr(chunk, "text"):
                documents.append(Document(text=chunk.text))
            else:
                raise TypeError(
                    f"Unsupported chunk type: {type(chunk)}"
                )

        self.logger.info("Chunks converted to Documents successfully")
        return documents



class VectorStoreManager:
    def __init__(self, base_path: str = "./vector_store"):
        self.base_path = Path(base_path)
        self.logger = loguru.logger
        self.model_name = "qwen3-embedding:4b"

        # discover embedding dimension dynamically
        
        self.embed_dim =2560


        # versioned persist dir
        self.vector_store_path = (
            self.base_path / self.model_name / f"dim_{self.embed_dim}"
        )

    def _exists(self) -> bool:
        return (
            (self.vector_store_path / "index.faiss").exists()
            and (self.vector_store_path / "docstore.json").exists()
        )
    def create_and_persist(self, documents: list) -> VectorStoreIndex:
        if not documents:
            raise ValueError("No documents provided for vector store creation")

        self.logger.info(
            f"Creating FAISS vector store ({self.model_name}, dim={self.embed_dim})"
        )

        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        faiss_index = faiss.IndexFlatL2(self.embed_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
           
        )

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

        index.storage_context.persist(
            persist_dir=str(self.vector_store_path)
        )

        return index

    def load_for_query(self) -> VectorStoreIndex:
        if not self._exists():
            raise RuntimeError(
                f"No vector index found for "
                f"{self.model_name} (dim={self.embed_dim})"
            )

        vector_store = FaissVectorStore.from_persist_dir(
            persist_dir=str(self.vector_store_path)
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(self.vector_store_path),
    )

        return load_index_from_storage(storage_context)

    def get_or_create(self, documents: list | None = None) -> VectorStoreIndex:
        if self._exists():
            return self.load_for_query()

        if documents is None:
            raise RuntimeError(
                "Vector store does not exist and no documents were provided"
            )

        return self.create_and_persist(documents)