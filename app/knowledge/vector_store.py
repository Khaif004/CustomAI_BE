import logging
from typing import List, Dict, Any, Optional
import os
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStoreManager:
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model
        )
        self.vector_store = self._initialize_store()
        self.collection_name = settings.vector_store_collection
        
    def _initialize_store(self):
        """Initialize vector store based on configuration"""
        try:
            os.makedirs(settings.vector_store_path, exist_ok=True)
            
            if settings.vector_store_type == "chroma":
                logger.info("Initializing Chroma vector store")
                store = Chroma(
                    collection_name=settings.vector_store_collection,
                    embedding_function=self.embeddings,
                    persist_directory=settings.vector_store_path
                )
            elif settings.vector_store_type == "faiss":
                logger.info("Initializing FAISS vector store")
                # FAISS will be initialized lazily on first use
                store = None
            else:
                raise ValueError(f"Unsupported vector store type: {settings.vector_store_type}")
            
            logger.info(f"Vector store initialized: {settings.vector_store_type}")
            return store
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            documents: List of Document objects
            metadata: Optional metadata to attach
            
        Returns:
            List of document IDs added
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return []
            
            # Add metadata to documents if provided
            if metadata:
                for doc in documents:
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata.update(metadata)
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            if settings.vector_store_type == "chroma":
                ids = self.vector_store.add_documents(documents)
            elif settings.vector_store_type == "faiss":
                # Initialize FAISS if not already done
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(
                        documents,
                        self.embeddings
                    )
                else:
                    # Add to existing FAISS store
                    ids = self.vector_store.add_documents(documents)
            
            logger.info(f"Successfully added {len(documents)} documents")
            return ids if ids else [f"doc_{i}" for i in range(len(documents))]
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Semantic search in vector store
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum relevance score
            
        Returns:
            List of search results with scores
        """
        try:
            logger.debug(f"Searching vector store: {query}")
            
            if settings.vector_store_type == "chroma":
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k
                )
            elif settings.vector_store_type == "faiss":
                if self.vector_store is None:
                    logger.warning("FAISS store not initialized, returning empty results")
                    return []
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k
                )
            
            # Filter by score threshold and format results
            formatted_results = []
            for doc, score in results:
                # Convert score to similarity (some implementations use distance)
                similarity = 1 / (1 + score) if score > 1 else score
                
                if similarity >= score_threshold:
                    formatted_results.append({
                        "content": doc.page_content,
                        "score": similarity,
                        "metadata": doc.metadata or {}
                    })
            
            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents from vector store"""
        try:
            logger.info(f"Deleting {len(ids)} documents from vector store")
            
            if settings.vector_store_type == "chroma":
                self.vector_store.delete(ids)
            elif settings.vector_store_type == "faiss":
                # FAISS doesn't support delete, would need to rebuild
                logger.warning("FAISS does not support document deletion")
                return False
            
            logger.info("Documents deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def update(self, documents: List[Document]) -> List[str]:
        """Update existing documents"""
        try:
            logger.info(f"Updating {len(documents)} documents")
            
            # For Chroma, documents are updated automatically if ID exists
            if settings.vector_store_type == "chroma":
                ids = self.vector_store.add_documents(documents)
            elif settings.vector_store_type == "faiss":
                # FAISS doesn't support updates, would need to rebuild
                logger.warning("FAISS does not support document updates")
                return []
            
            return ids
            
        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            raise
    
    def persist(self):
        """Persist vector store to disk"""
        try:
            if settings.vector_store_type == "chroma":
                self.vector_store.persist()
                logger.info("Vector store persisted")
            elif settings.vector_store_type == "faiss":
                if self.vector_store:
                    self.vector_store.save_local(settings.vector_store_path)
                    logger.info("FAISS store saved")
                    
        except Exception as e:
            logger.error(f"Error persisting vector store: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            if settings.vector_store_type == "chroma":
                # Get collection size
                collection = self.vector_store._collection
                stats = {
                    "type": "chroma",
                    "collection_name": self.collection_name,
                    "total_documents": collection.count() if hasattr(collection, 'count') else 0,
                }
            elif settings.vector_store_type == "faiss":
                stats = {
                    "type": "faiss",
                    "total_documents": self.vector_store.index.ntotal if self.vector_store else 0,
                }
            else:
                stats = {}
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error getting vector store stats: {str(e)}")
            return {"error": str(e)}
