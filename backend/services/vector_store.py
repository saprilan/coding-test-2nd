from typing import List, Tuple
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from config import settings
import logging
import os

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = OllamaEmbeddings(model="mistral")
        
        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            persist_directory=settings.vector_db_path,
            embedding_function=self.embedding_model
        )
        logger.info("Vector store initialized at %s", settings.vector_db_path)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        try:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            logger.info("Added %d documents to vector store", len(documents))
        except Exception as e:
            logger.error("Error adding documents: %s", str(e))

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info("Similarity search returned %d results", len(results))
            return results
        except Exception as e:
            logger.error("Error during similarity search: %s", str(e))
            return []

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from vector store"""
        try:
            self.vectorstore.delete(ids=document_ids)
            self.vectorstore.persist()
            logger.info("Deleted documents: %s", document_ids)
        except Exception as e:
            logger.error("Error deleting documents: %s", str(e))

    def get_document_count(self) -> int:
        """Get total number of documents in vector store"""
        try:
            collection = self.vectorstore._collection  # Accessing Chroma collection directly
            count = collection.count()
            logger.info("Document count: %d", count)
            return count
        except Exception as e:
            logger.error("Error retrieving document count: %s", str(e))
            return 0
