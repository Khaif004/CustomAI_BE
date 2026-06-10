import logging
import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SAPAICoreEmbeddings:
    """
    LangChain-compatible Embeddings class that calls SAP AI Core's
    OpenAI-compatible embeddings endpoint:
      POST /v2/inference/deployments/{id}/embeddings
    with an OAuth2 Bearer token.
    """

    def __init__(self, aicore_url: str, auth_url: str, client_id: str,
                 client_secret: str, deployment_id: str, model: str):
        import httpx
        self._httpx = httpx
        # SAP AI Core foundation-models: correct inference path is /v1/embeddings (not /embeddings).
        base = aicore_url.rstrip('/')
        self.embeddings_url = f"{base}/v2/inference/deployments/{deployment_id}/v1/embeddings"
        self.token_url = (
            auth_url if auth_url.endswith("/oauth/token")
            else f"{auth_url}/oauth/token"
        )
        self.client_id = client_id
        self.client_secret = client_secret
        self.model = model
        self._token = None
        self._token_expiry = 0
        logger.info(f"SAP AI Core Embeddings — deployment: {deployment_id}, model: {model}")

    def _get_token(self) -> str:
        import time
        if self._token and time.time() < self._token_expiry:
            return self._token
        resp = self._httpx.post(
            self.token_url,
            data={"client_id": self.client_id, "client_secret": self.client_secret,
                  "grant_type": "client_credentials"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600) - 60
        return self._token

    def _embed(self, texts: list, batch_size: int = 20) -> list:
        """Embed texts in batches — one API call per batch instead of per text."""
        token = self._get_token()
        results = []
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": "default",
        }
        for i in range(0, len(texts), batch_size):
            batch = [t[:4000] if len(t) > 4000 else t for t in texts[i:i + batch_size]]
            # /v1/embeddings accepts an array of strings (OpenAI-compatible)
            resp = self._httpx.post(
                self.embeddings_url,
                headers=headers,
                json={"input": batch},
                timeout=120,
            )
            if not resp.is_success:
                logger.error(
                    f"SAP AI Core embeddings {resp.status_code}: {resp.text[:500]}"
                )
                resp.raise_for_status()
            data = resp.json()
            # Sort by index to preserve order
            items = sorted(data["data"], key=lambda x: x.get("index", 0))
            results.extend(item["embedding"] for item in items)
            logger.debug(f"Embedded batch {i // batch_size + 1}: {len(batch)} texts")
        return results

    def embed_documents(self, texts: list) -> list:
        return self._embed(texts)

    def embed_query(self, text: str) -> list:
        return self._embed([text])[0]


def _build_embeddings():
    """Build the embeddings object based on the configured LLM provider."""
    if settings.llm_provider == "sap_ai_core" and settings.sap_aicore_url:
        embedding_deployment = (
            settings.sap_aicore_embedding_deployment_id or settings.sap_aicore_deployment_id
        )
        return SAPAICoreEmbeddings(
            aicore_url=settings.sap_aicore_url,
            auth_url=settings.sap_aicore_auth_url,
            client_id=settings.sap_aicore_client_id,
            client_secret=settings.sap_aicore_client_secret,
            deployment_id=embedding_deployment,
            model=settings.embedding_model,
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Using OpenAI embeddings — model: {settings.embedding_model}")
        return OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )


class VectorStoreManager:

    def __init__(self):
        self.embeddings = _build_embeddings()
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

    def search(self, query: str, k: int = 5, score_threshold: float = 0.0,
               metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if settings.vector_store_type == "faiss" and self.vector_store is None:
            return []

        try:
            if metadata_filter and settings.vector_store_type == "chroma":
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=metadata_filter
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.warning(f"Vector search failed (filter={metadata_filter}): {e}")
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
