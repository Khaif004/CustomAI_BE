import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from app.models.chat import Conversation, ConversationStatus, ChatMessage

logger = logging.getLogger(__name__)


class DocumentStore:
    """In-memory conversation store with file persistence"""

    def __init__(self, storage_path: str = "./data/document_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._conversations_cache: Dict[str, Conversation] = {}
        self._load_cache()

    def _load_cache(self):
        conversations_dir = self.storage_path / "conversations"
        if conversations_dir.exists():
            for file in conversations_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        conv = Conversation(**json.load(f))
                        self._conversations_cache[conv.id] = conv
                except Exception as e:
                    logger.warning(f"Error loading conversation {file.name}: {e}")
        logger.info(f"Loaded {len(self._conversations_cache)} conversations")

    def _save_conversation_to_disk(self, conversation: Conversation):
        try:
            conversations_dir = self.storage_path / "conversations"
            conversations_dir.mkdir(parents=True, exist_ok=True)
            with open(conversations_dir / f"{conversation.id}.json", 'w') as f:
                json.dump(conversation.model_dump(mode='json'), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

    def create_conversation(self, title: Optional[str] = None, user_id: Optional[str] = None,
                            project_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Conversation:
        import uuid
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        conversation = Conversation(id=conv_id, title=title or f"Conversation {conv_id}",
                                     user_id=user_id, project_id=project_id, metadata=metadata or {})
        self._conversations_cache[conv_id] = conversation
        self._save_conversation_to_disk(conversation)
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations_cache.get(conversation_id)

    def add_message(self, conversation_id: str, message: ChatMessage) -> bool:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        conversation.messages.append(message)
        conversation.updated_at = datetime.utcnow()
        self._save_conversation_to_disk(conversation)
        return True

    def get_conversations(self, user_id: Optional[str] = None, project_id: Optional[str] = None,
                          status: Optional[ConversationStatus] = None, limit: int = 50, offset: int = 0) -> List[Conversation]:
        conversations = list(self._conversations_cache.values())
        if user_id:
            conversations = [c for c in conversations if c.user_id == user_id]
        if project_id:
            conversations = [c for c in conversations if c.project_id == project_id]
        if status:
            conversations = [c for c in conversations if c.status == status]
        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        return conversations[offset:offset + limit]

    def update_conversation_status(self, conversation_id: str, status: ConversationStatus) -> bool:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        conversation.status = status
        conversation.updated_at = datetime.utcnow()
        self._save_conversation_to_disk(conversation)
        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id not in self._conversations_cache:
            return False
        del self._conversations_cache[conversation_id]
        file_path = self.storage_path / "conversations" / f"{conversation_id}.json"
        if file_path.exists():
            file_path.unlink()
        return True

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._conversations_cache)
        active = sum(1 for c in self._conversations_cache.values() if c.status == ConversationStatus.ACTIVE)
        messages = sum(len(c.messages) for c in self._conversations_cache.values())
        return {"total_conversations": total, "active_conversations": active,
                "completed_conversations": total - active, "total_messages": messages}

    def cleanup(self):
        for conversation in self._conversations_cache.values():
            self._save_conversation_to_disk(conversation)
