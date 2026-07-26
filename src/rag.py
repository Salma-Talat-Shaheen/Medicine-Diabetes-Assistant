"""RAG (Retrieval Augmented Generation) components for the Medicine Assistant."""

import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config import settings


class RAGComponent:
    """Handles retrieval from Chroma vector store for RAG."""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the RAG component for retrieval only.

        Args:
            persist_directory: Directory where the vector store is persisted.
                             Defaults to settings.CHROMA_PERSIST_DIRECTORY.
        """
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        settings.validate()
        
        self.embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
        self._vector_store: Optional[Chroma] = None

    @property
    def vector_store(self) -> Optional[Chroma]:
        """Get or create the vector store safely."""
        if self._vector_store is None:
            if not os.path.exists(self.persist_directory):
                print(f"[RAG Warning] Chroma persist directory not found at: {self.persist_directory}")
                return None
            try:
                self._vector_store = Chroma(
                    collection_name=settings.COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            except Exception as e:
                print(f"[RAG Error] Failed to load Chroma vector store: {e}")
                return None
        return self._vector_store

    def get_retriever(self) -> Optional[VectorStoreRetriever]:
        """
        Get a retriever for the vector store.

        Returns:
            VectorStoreRetriever configured with current settings or None.
        """
        vs = self.vector_store
        if vs is None:
            return None
        return vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K_RESULTS},
        )

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query.
            k: Optional number of results to return.

        Returns:
            List of relevant documents or empty list if vector store is unavailable.
        """
        vs = self.vector_store
        if vs is None:
            print("[RAG Warning] Vector store is not available. Returning empty results.")
            return []
            
        top_k = k if (k is not None) else settings.TOP_K_RESULTS
        try:
            return vs.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"[RAG Error] Error during similarity search: {e}")
            return []
