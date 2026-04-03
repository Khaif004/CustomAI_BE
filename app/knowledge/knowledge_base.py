import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document
from app.knowledge.vector_store import VectorStoreManager
from app.knowledge.document_store import DocumentStore
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class KnowledgeBaseManager:
    """Central manager for document ingestion, chunking, and semantic search"""

    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.document_store = DocumentStore()
        self._chunk_size = settings.knowledge_base_chunk_size
        self._chunk_overlap = settings.knowledge_base_chunk_overlap
        logger.info("Knowledge Base Manager initialized")

    def ingest_documents(self, file_paths: List[str], category: str = "general",
                         source: Optional[str] = None) -> Dict[str, Any]:
        documents = []
        for file_path in file_paths:
            try:
                documents.extend(self._load_file(file_path, category, source))
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        if documents:
            self.vector_store.add_documents(documents)

        return {"status": "success", "files_processed": len(file_paths), "total_chunks": len(documents)}

    def _load_file(self, file_path: str, category: str, source: Optional[str]) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix not in (".txt", ".md", ".py"):
            raise ValueError(f"Unsupported file type: {path.suffix}")

        content = path.read_text(encoding='utf-8')
        chunks = self._chunk_text(content, self._chunk_size, self._chunk_overlap)

        return [
            Document(page_content=chunk, metadata={
                "source": source or str(file_path), "filename": path.name,
                "category": category, "chunk_index": i, "total_chunks": len(chunks)
            })
            for i, chunk in enumerate(chunks)
        ]

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return chunks

    def search(self, query: str, k: int = 5, score_threshold: float = 0.0,
               category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            results = self.vector_store.search(query, k=k, score_threshold=score_threshold)
            if category_filter:
                results = [r for r in results if r.get("metadata", {}).get("category") == category_filter]
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_knowledge_stats(self) -> Dict[str, Any]:
        try:
            vector_stats = self.vector_store.get_stats()
            doc_stats = self.document_store.get_stats()
            return {"vector_store": vector_stats, "document_store": doc_stats,
                    "total_documents": vector_stats.get("total_documents", 0),
                    "total_conversations": doc_stats.get("total_conversations", 0)}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def persist(self):
        self.vector_store.persist()
        self.document_store.cleanup()


_kb_manager: Optional[KnowledgeBaseManager] = None


def get_knowledge_base() -> KnowledgeBaseManager:
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    return _kb_manager
