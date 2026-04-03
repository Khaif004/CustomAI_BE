import logging
import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStoreManager:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key, model=settings.embedding_model)
        self.collection_name = settings.vector_store_collection
        self.vector_store = self._initialize_store()

    def _initialize_store(self):
        os.makedirs(settings.vector_store_path, exist_ok=True)

        if settings.vector_store_type == "chroma":
            store = Chroma(collection_name=self.collection_name, embedding_function=self.embeddings,
                           persist_directory=settings.vector_store_path)
        elif settings.vector_store_type == "faiss":
            store = None  # Initialized lazily on first document add
        else:
            raise ValueError(f"Unsupported vector store type: {settings.vector_store_type}")

        logger.info(f"Vector store initialized: {settings.vector_store_type}")
        return store

    def add_documents(self, documents: List[Document], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        if not documents:
            return []

        if metadata:
            for doc in documents:
                doc.metadata = {**(doc.metadata or {}), **metadata}

        if settings.vector_store_type == "faiss" and self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            return [f"doc_{i}" for i in range(len(documents))]

        ids = self.vector_store.add_documents(documents)
        return ids or [f"doc_{i}" for i in range(len(documents))]

    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        if settings.vector_store_type == "faiss" and self.vector_store is None:
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)

        return [
            {"content": doc.page_content, "score": 1 / (1 + score) if score > 1 else score,
             "metadata": doc.metadata or {}}
            for doc, score in results
            if (1 / (1 + score) if score > 1 else score) >= score_threshold
        ]

    def delete(self, ids: List[str]) -> bool:
        if settings.vector_store_type == "chroma":
            self.vector_store.delete(ids)
            return True
        return False

    def persist(self):
        if settings.vector_store_type == "chroma":
            self.vector_store.persist()
        elif settings.vector_store_type == "faiss" and self.vector_store:
            self.vector_store.save_local(settings.vector_store_path)

    def get_stats(self) -> Dict[str, Any]:
        try:
            if settings.vector_store_type == "chroma":
                collection = self.vector_store._collection
                return {"type": "chroma", "collection_name": self.collection_name,
                        "total_documents": collection.count() if hasattr(collection, 'count') else 0}
            elif settings.vector_store_type == "faiss":
                return {"type": "faiss",
                        "total_documents": self.vector_store.index.ntotal if self.vector_store else 0}
            return {}
        except Exception as e:
            return {"error": str(e)}
