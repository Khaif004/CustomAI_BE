"""
Document Store Management

Manages persistent storage of conversations, metadata, and system data.
Currently uses in-memory storage with file-based fallback.
In production, should be replaced with PostgreSQL.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from app.models.chat import Conversation, ConversationStatus, ChatMessage

logger = logging.getLogger(__name__)


class DocumentStore:
    """Manages persistent storage of conversations and metadata"""
    
    def __init__(self, storage_path: str = "./data/document_store"):
        """
        Initialize document store
        
        Args:
            storage_path: Path to store documents
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._conversations_cache: Dict[str, Conversation] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load existing conversations from disk"""
        try:
            conversations_dir = self.storage_path / "conversations"
            if conversations_dir.exists():
                for file in conversations_dir.glob("*.json"):
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                            conv = Conversation(**data)
                            self._conversations_cache[conv.id] = conv
                        logger.debug(f"Loaded conversation: {file.name}")
                    except Exception as e:
                        logger.warning(f"Error loading conversation {file.name}: {str(e)}")
            
            logger.info(f"Loaded {len(self._conversations_cache)} conversations from cache")
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
    
    def _save_conversation_to_disk(self, conversation: Conversation):
        """Save single conversation to disk"""
        try:
            conversations_dir = self.storage_path / "conversations"
            conversations_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = conversations_dir / f"{conversation.id}.json"
            with open(file_path, 'w') as f:
                json.dump(
                    conversation.model_dump(mode='json'),
                    f,
                    indent=2,
                    default=str
                )
            
        except Exception as e:
            logger.error(f"Error saving conversation to disk: {str(e)}")
    
    # ==================== Conversation Operations ====================
    
    def create_conversation(
        self,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create new conversation"""
        try:
            import uuid
            conv_id = f"conv_{uuid.uuid4().hex[:12]}"
            
            conversation = Conversation(
                id=conv_id,
                title=title or f"Conversation {conv_id}",
                user_id=user_id,
                project_id=project_id,
                metadata=metadata or {}
            )
            
            self._conversations_cache[conv_id] = conversation
            self._save_conversation_to_disk(conversation)
            
            logger.info(f"Created conversation: {conv_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        try:
            return self._conversations_cache.get(conversation_id)
        except Exception as e:
            logger.error(f"Error getting conversation: {str(e)}")
            return None
    
    def add_message(self, conversation_id: str, message: ChatMessage) -> bool:
        """Add message to conversation"""
        try:
            conversation = self.get_conversation(conversation_id)
            if not conversation:
                logger.warning(f"Conversation not found: {conversation_id}")
                return False
            
            conversation.messages.append(message)
            conversation.updated_at = datetime.utcnow()
            
            self._save_conversation_to_disk(conversation)
            logger.debug(f"Added message to conversation: {conversation_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return False
    
    def get_conversations(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        status: Optional[ConversationStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """Get conversations with filtering"""
        try:
            conversations = list(self._conversations_cache.values())
            
            # Apply filters
            if user_id:
                conversations = [c for c in conversations if c.user_id == user_id]
            if project_id:
                conversations = [c for c in conversations if c.project_id == project_id]
            if status:
                conversations = [c for c in conversations if c.status == status]
            
            # Sort by updated_at descending
            conversations.sort(key=lambda c: c.updated_at, reverse=True)
            
            # Apply pagination
            return conversations[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Error getting conversations: {str(e)}")
            return []
    
    def update_conversation_status(
        self,
        conversation_id: str,
        status: ConversationStatus
    ) -> bool:
        """Update conversation status"""
        try:
            conversation = self.get_conversation(conversation_id)
            if not conversation:
                return False
            
            conversation.status = status
            conversation.updated_at = datetime.utcnow()
            
            self._save_conversation_to_disk(conversation)
            logger.info(f"Updated conversation status: {conversation_id} -> {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating conversation status: {str(e)}")
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation"""
        try:
            if conversation_id in self._conversations_cache:
                del self._conversations_cache[conversation_id]
                
                # Delete from disk
                conversations_dir = self.storage_path / "conversations"
                file_path = conversations_dir / f"{conversation_id}.json"
                if file_path.exists():
                    file_path.unlink()
                
                logger.info(f"Deleted conversation: {conversation_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            return False
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics"""
        try:
            total_conversations = len(self._conversations_cache)
            active_conversations = sum(
                1 for c in self._conversations_cache.values()
                if c.status == ConversationStatus.ACTIVE
            )
            total_messages = sum(
                len(c.messages) for c in self._conversations_cache.values()
            )
            
            return {
                "total_conversations": total_conversations,
                "active_conversations": active_conversations,
                "completed_conversations": total_conversations - active_conversations,
                "total_messages": total_messages,
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Save all conversations before cleanup
            for conversation in self._conversations_cache.values():
                self._save_conversation_to_disk(conversation)
            
            logger.info("Document store cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
