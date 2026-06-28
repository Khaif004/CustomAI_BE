"""
PgVector-based vector store backed by Neon.tech PostgreSQL.

Replaces ChromaDB with a persistent, enterprise-grade vector store.
Documents go into `knowledge_documents`; embeddings into `embeddings`.
Both tables live in the existing Neon schema.
"""
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
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
                 client_secret: str, deployment_id: str, model: str,
                 resource_group: str = "default"):
        import httpx
        self._httpx = httpx
        base = aicore_url.rstrip('/')
        self.embeddings_url = f"{base}/v2/inference/deployments/{deployment_id}/v1/embeddings"
        self.token_url = (
            auth_url if auth_url.endswith("/oauth/token")
            else f"{auth_url}/oauth/token"
        )
        self.client_id = client_id
        self.client_secret = client_secret
        self.model = model
        self._resource_group = resource_group or "default"
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
            "AI-Resource-Group": self._resource_group,
        }
        for i in range(0, len(texts), batch_size):
            batch = [t[:4000] if len(t) > 4000 else t for t in texts[i:i + batch_size]]
            resp = self._httpx.post(
                self.embeddings_url,
                headers=headers,
                json={"input": batch},
                timeout=120,
            )
            if not resp.is_success:
                if resp.status_code == 404:
                    raise RuntimeError(
                        f"Embedding deployment not found (404). "
                        f"Set SAP_AICORE_EMBEDDING_DEPLOYMENT_ID in .env to a valid deployment. "
                        f"URL: {self.embeddings_url}"
                    )
                logger.error(f"SAP AI Core embeddings {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
            data = resp.json()
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
            resource_group=getattr(settings, "sap_aicore_resource_group", "default"),
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Using OpenAI embeddings — model: {settings.embedding_model}")
        return OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )


class VectorStoreManager:
    """
    pgvector-backed vector store using Neon.tech PostgreSQL.

    Public interface is identical to the old ChromaDB-based manager so
    KnowledgeBaseManager and search callers need no changes.
    """

    def __init__(self):
        self.embeddings = _build_embeddings()
        self.collection_name = settings.vector_store_collection

        if not settings.neon_db_url:
            raise RuntimeError(
                "NEON_DB_URL environment variable is required. "
                "Set it to your Neon PostgreSQL connection string in .env."
            )
        self._db_url = settings.neon_db_url
        self._verify_connection()

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _connect(self):
        import psycopg2
        from pgvector.psycopg2 import register_vector
        conn = psycopg2.connect(self._db_url)
        register_vector(conn)
        return conn

    def _verify_connection(self):
        try:
            conn = self._connect()
            conn.close()
            logger.info("VectorStoreManager — Neon pgvector connection verified.")
        except Exception as e:
            logger.error(f"VectorStoreManager — cannot connect to Neon: {e}")

    def _get_or_create_app_uuid(self, cur, app_id: str, app_name: str = "") -> str:
        """Upsert the application row and return its UUID."""
        cur.execute(
            """
            INSERT INTO applications (application_key, name)
            VALUES (%s, %s)
            ON CONFLICT (application_key)
            DO UPDATE SET name = EXCLUDED.name, updated_at = NOW()
            RETURNING id
            """,
            (app_id, app_name or app_id),
        )
        return str(cur.fetchone()[0])

    # ── Public API ──────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if not documents:
            return []

        if metadata:
            for doc in documents:
                doc.metadata = {**(doc.metadata or {}), **metadata}

        # Extract app identity from the first document
        # Documents without an explicit app_id are stored under "__global__"
        # (general knowledge that applies to all apps)
        first_meta = documents[0].metadata or {}
        app_id: str = first_meta.get("app_id", "") or "__global__"
        app_name: str = first_meta.get("app_name", "") or "Global Knowledge Base"

        texts = [doc.page_content for doc in documents]
        try:
            vectors = self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Embedding failed for app '{app_id}': {e}")
            return []

        ids: List[str] = []
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    app_uuid = self._get_or_create_app_uuid(cur, app_id, app_name) if app_id else None

                    for doc, vector in zip(documents, vectors):
                        meta = doc.metadata or {}
                        doc_type = meta.get("category", "schema")
                        title = meta.get("title", "")
                        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()[:64]

                        cur.execute(
                            """
                            INSERT INTO knowledge_documents
                                (application_id, document_type, title, content, metadata, content_hash)
                            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                            RETURNING id
                            """,
                            (app_uuid, doc_type, title, doc.page_content,
                             json.dumps(meta), content_hash),
                        )
                        doc_uuid = str(cur.fetchone()[0])

                        cur.execute(
                            """
                            INSERT INTO embeddings
                                (application_id, document_id, document_type, content, embedding, model_version)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (app_uuid, doc_uuid, doc_type, doc.page_content,
                             vector, settings.embedding_model),
                        )
                        ids.append(str(cur.fetchone()[0]))

        finally:
            conn.close()

        logger.info(f"Stored {len(ids)} embeddings for app '{app_id}'")
        return ids

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            query_vector = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        # pgvector's psycopg2 adapter registers a type encoder for numpy.ndarray
        # (not for plain Python lists).  Plain lists are serialized as numeric[]
        # which lacks the <=> operator.  Convert to numpy so the encoder fires.
        try:
            import numpy as np
            query_vector = np.array(query_vector, dtype=np.float32)
        except ImportError:
            pass  # numpy unavailable — fall back to ::vector cast in SQL below

        app_id: Optional[str] = None
        category: Optional[str] = None
        if metadata_filter:
            app_id = metadata_filter.get("app_id")
            category = metadata_filter.get("category")

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                if app_id:
                    cur.execute(
                        """
                        SELECT e.content,
                               1 - (e.embedding <=> %s::vector) AS score,
                               kd.title, kd.document_type, kd.metadata
                        FROM embeddings e
                        JOIN knowledge_documents kd ON e.document_id = kd.id
                        JOIN applications a ON e.application_id = a.id
                        WHERE a.application_key = %s
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_vector, app_id, query_vector, k),
                    )
                elif category:
                    cur.execute(
                        """
                        SELECT e.content,
                               1 - (e.embedding <=> %s::vector) AS score,
                               kd.title, kd.document_type, kd.metadata
                        FROM embeddings e
                        JOIN knowledge_documents kd ON e.document_id = kd.id
                        WHERE kd.document_type = %s
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_vector, category, query_vector, k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT e.content,
                               1 - (e.embedding <=> %s::vector) AS score,
                               kd.title, kd.document_type, kd.metadata
                        FROM embeddings e
                        JOIN knowledge_documents kd ON e.document_id = kd.id
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_vector, query_vector, k),
                    )

                rows = cur.fetchall()

        finally:
            conn.close()

        results: List[Dict[str, Any]] = []
        for content, score, title, doc_type, meta_json in rows:
            score_f = float(score)
            if score_f < score_threshold:
                continue
            meta = meta_json if isinstance(meta_json, dict) else {}
            if title:
                meta["title"] = title
            if doc_type:
                meta["category"] = doc_type
            results.append({"content": content, "score": score_f, "metadata": meta})

        return results

    def delete(self, app_id: str) -> bool:
        """Delete all documents and embeddings for an app."""
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM knowledge_documents
                        WHERE application_id = (
                            SELECT id FROM applications WHERE application_key = %s
                        )
                        """,
                        (app_id,),
                    )
                    count = cur.rowcount
            logger.info(f"Deleted {count} documents for app '{app_id}'")
            return True
        except Exception as e:
            logger.error(f"Delete failed for app '{app_id}': {e}")
            return False
        finally:
            conn.close()

    def persist(self):
        pass  # Neon auto-persists every write — nothing to do

    def get_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM embeddings")
                total = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT application_id) FROM embeddings")
                apps = cur.fetchone()[0]
            return {
                "type": "pgvector",
                "collection_name": self.collection_name,
                "total_documents": int(total),
                "total_apps": int(apps),
            }
        except Exception as e:
            return {"type": "pgvector", "error": str(e)}
        finally:
            conn.close()
