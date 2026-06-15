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


    def register_app_context(
        self,
        app_id: str,
        app_name: str,
        documents: List[Dict[str, str]],
        replace: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest plain-text context documents for a specific app.

        documents: list of {"title": str, "content": str}
        replace:   if True, delete existing chunks for this app_id first.
        """
        if replace:
            self._delete_by_app_id(app_id)

        chunks: List[Document] = []
        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            if not content.strip():
                continue
            for i, chunk in enumerate(self._chunk_text(content, self._chunk_size, self._chunk_overlap)):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "app_id": app_id,
                        "app_name": app_name,
                        "title": title,
                        "category": f"app:{app_id}",
                        "chunk_index": i,
                    }
                ))

        if chunks:
            self.vector_store.add_documents(chunks)

        logger.info(f"Registered {len(chunks)} chunks for app '{app_id}'")
        return {"app_id": app_id, "chunks_stored": len(chunks), "docs_received": len(documents)}

    def _delete_by_app_id(self, app_id: str):
        """Delete all stored chunks for a given app_id (Chroma only)."""
        try:
            if settings.vector_store_type == "chroma":
                col = self.vector_store.vector_store._collection
                results = col.get(where={"app_id": app_id})
                ids = results.get("ids", [])
                if ids:
                    col.delete(ids=ids)
                    logger.info(f"Deleted {len(ids)} old chunks for app '{app_id}'")
        except Exception as e:
            logger.warning(f"Could not delete old chunks for app '{app_id}': {e}")

    def search_with_app_context(
        self,
        query: str,
        app_id: Optional[str] = None,
        k_app: int = 8,
        k_global: int = 3,
        score_threshold: float = 0.0,
    ) -> str:
        """
        Search vector store and return a formatted RAG context string.

        - If app_id given: fetch k_app app-scoped chunks + k_global global chunks.
        - If no app_id: fetch k_global+k_app global chunks only.
        Returns empty string when nothing relevant is found.
        """
        all_results: List[Dict[str, Any]] = []

        if app_id:
            app_results = self.vector_store.search(
                query, k=k_app, score_threshold=score_threshold,
                metadata_filter={"app_id": app_id}
            )
            all_results.extend(app_results)

        global_results = self.vector_store.search(
            query, k=k_global, score_threshold=score_threshold,
            metadata_filter={"category": "general"}
        )
        all_results.extend(global_results)

        if not all_results:
            return ""

        lines = ["Relevant knowledge retrieved from the knowledge base:"]
        for r in all_results:
            meta = r.get("metadata", {})
            source = meta.get("title") or meta.get("app_name") or meta.get("source", "")
            label = f"[{source}] " if source else ""
            lines.append(f"\n{label}{r['content']}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Original file-based ingestion (unchanged)                           #
    # ------------------------------------------------------------------ #

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
