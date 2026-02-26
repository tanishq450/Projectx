from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
import fitz
import os
import loguru
from chonkie import TokenChunker
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from pathlib import Path
from llama_index.core import load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import chromadb 
from chroma_client import get_chroma_client



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
    def __init__(self, chunk_size: int = 1000, output_dir: str = "./data",stride: int = 200):
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.logger = loguru.logger
        self.stride = stride
        
    def chunk_text(self,text:str):
        try:
            self.logger.info(f"Chunking text")
            chunker = TokenChunker(chunk_size=self.chunk_size,chunk_overlap=self.stride)
            chunks = chunker(text)
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
    def __init__(self):
        self.logger = loguru.logger

    def create(self, documents: list, persist_dir: str) -> VectorStoreIndex:
        if not documents:
            raise ValueError("No documents provided")

        self.logger.info(f"Creating Chroma index at {persist_dir}")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        chroma_client = get_chroma_client(persist_dir)

        collection = chroma_client.get_or_create_collection(
            name="rag_docs"
        )

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

        storage_context.persist(persist_dir=persist_dir)
        return index

    def load(self, persist_dir: str) -> VectorStoreIndex:
        if not Path(persist_dir).exists():
            raise RuntimeError(f"No vector index found at {persist_dir}")

        chroma_client = get_chroma_client(persist_dir)

        collection = chroma_client.get_or_create_collection(
            name="rag_docs"
        )

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )