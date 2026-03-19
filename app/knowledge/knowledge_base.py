"""
Knowledge Base Manager

Central coordinator for all knowledge base operations.
Manages vector store, document store, and knowledge ingestion.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.schema import Document
from app.knowledge.vector_store import VectorStoreManager
from app.knowledge.document_store import DocumentStore
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class KnowledgeBaseManager:
    """Central manager for knowledge base operations"""
    
    def __init__(self):
        """Initialize knowledge base components"""
        try:
            self.vector_store = VectorStoreManager()
            self.document_store = DocumentStore()
            self._chunk_size = settings.knowledge_base_chunk_size
            self._chunk_overlap = settings.knowledge_base_chunk_overlap
            
            logger.info("Knowledge Base Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Base Manager: {str(e)}")
            raise
    
    def ingest_documents(
        self,
        file_paths: List[str],
        category: str = "general",
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest documents from files
        
        Args:
            file_paths: List of file paths to ingest
            category: Category for documents
            source: Source identifier
            
        Returns:
            Ingestion results
        """
        try:
            logger.info(f"Starting document ingestion: {len(file_paths)} files")
            
            documents = []
            total_chunks = 0
            
            for file_path in file_paths:
                try:
                    file_documents = self._load_file(file_path, category, source)
                    documents.extend(file_documents)
                    logger.debug(f"Loaded {len(file_documents)} chunks from {file_path}")
                    
                except Exception as e:
                    logger.warning(f"Error loading file {file_path}: {str(e)}")
                    continue
            
            if documents:
                doc_ids = self.vector_store.add_documents(documents)
                total_chunks = len(documents)
                logger.info(f"Ingested {total_chunks} document chunks")
            
            return {
                "status": "success",
                "files_processed": len(file_paths),
                "total_chunks": total_chunks,
                "documents_added": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _load_file(
        self,
        file_path: str,
        category: str,
        source: Optional[str]
    ) -> List[Document]:
        """
        Load and chunk file content
        
        Args:
            file_path: Path to file
            category: Document category
            source: Source identifier
            
        Returns:
            List of chunked documents
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file based on extension
        if path.suffix == ".txt":
            content = path.read_text(encoding='utf-8')
        elif path.suffix == ".md":
            content = path.read_text(encoding='utf-8')
        elif path.suffix == ".py":
            content = path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Chunk content
        chunks = self._chunk_text(content, self._chunk_size, self._chunk_overlap)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source or str(file_path),
                    "filename": path.name,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum relevance score
            category_filter: Optional category filter
            
        Returns:
            Search results
        """
        try:
            results = self.vector_store.search(query, k=k, score_threshold=score_threshold)
            
            # Apply category filter if provided
            if category_filter:
                results = [
                    r for r in results
                    if r.get("metadata", {}).get("category") == category_filter
                ]
            
            logger.debug(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            vector_stats = self.vector_store.get_stats()
            doc_stats = self.document_store.get_stats()
            
            return {
                "vector_store": vector_stats,
                "document_store": doc_stats,
                "total_documents": vector_stats.get("total_documents", 0),
                "total_conversations": doc_stats.get("total_conversations", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def persist(self):
        """Persist all knowledge base data"""
        try:
            self.vector_store.persist()
            self.document_store.cleanup()
            logger.info("Knowledge base persisted")
            
        except Exception as e:
            logger.error(f"Error persisting knowledge base: {str(e)}")
            raise


# Global instance (lazy initialization)
_kb_manager: Optional[KnowledgeBaseManager] = None


def get_knowledge_base() -> KnowledgeBaseManager:
    """Get or initialize knowledge base manager"""
    global _kb_manager
    
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    
    return _kb_manager
