import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.models.chat import AgentType, ChatMessage
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.specialized_agents import AgentFactory
from app.knowledge.document_store import DocumentStore

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents for query processing"""

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.agent_factory = AgentFactory()
        self.document_store = DocumentStore()
        self.total_queries = 0
        self.total_processing_time = 0
        self.agent_calls = {}
        logger.info("Multi-Agent Orchestrator initialized")

    async def process_query(self, query: str, conversation_id: Optional[str] = None,
                            user_id: Optional[str] = None, project_context: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        try:
            self.total_queries += 1

            # Get or create conversation
            if not conversation_id:
                conversation = self.document_store.create_conversation(user_id=user_id, project_id=project_context)
                conversation_id = conversation.id
            else:
                conversation = self.document_store.get_conversation(conversation_id)
                if not conversation:
                    return {"status": "error", "message": "Conversation not found"}

            # Add user message
            user_message = ChatMessage(role="user", content=query, timestamp=datetime.utcnow(),
                                       metadata={"project_context": project_context})
            self.document_store.add_message(conversation_id, user_message)

            conversation_history = [{"role": m.role, "content": m.content} for m in conversation.messages[:-1]]

            # Route through supervisor
            supervision_result = await self.supervisor.process_query(
                query, context={"project": project_context}, conversation_history=conversation_history)
            selected_agents = supervision_result.get("selected_agents", ["developer_helper"])

            # Process with selected agents
            agent_responses = await self._process_with_agents(query, selected_agents, conversation_history)

            # Aggregate results
            final_response = await self._aggregate_responses(query, agent_responses, supervision_result)

            # Save assistant message
            assistant_message = ChatMessage(
                role="assistant", content=final_response.get("response", ""),
                timestamp=datetime.utcnow(), agent_type=AgentType.SUPERVISOR,
                metadata={"agents_used": selected_agents, "routing_decision": supervision_result.get("routing_decision")})
            self.document_store.add_message(conversation_id, assistant_message)

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            final_response["conversation_id"] = conversation_id
            final_response["processing_time"] = processing_time
            final_response["timestamp"] = datetime.utcnow().isoformat()

            logger.info(f"Query processed in {processing_time:.2f}s")
            return final_response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"status": "error", "message": str(e), "conversation_id": conversation_id,
                    "processing_time": time.time() - start_time}

    async def _process_with_agents(self, query: str, agent_names: List[str],
                                    conversation_history: List[Dict]) -> Dict[str, Any]:
        """Process query with selected agents in parallel"""
        responses = {}
        tasks = []
        agent_map = {}

        for agent_name in agent_names:
            try:
                agent_type = self._map_agent_name_to_type(agent_name)
                agent = self.agent_factory.get_agent(agent_type)
                task = agent.process_query(query, context={"agent_name": agent_name},
                                           conversation_history=conversation_history)
                tasks.append(task)
                agent_map[len(tasks) - 1] = agent_name
                self.agent_calls[agent_name] = self.agent_calls.get(agent_name, 0) + 1
            except Exception as e:
                logger.warning(f"Error initializing agent {agent_name}: {e}")
                responses[agent_name] = {"status": "error", "error": str(e)}

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                agent_name = agent_map.get(idx)
                if agent_name:
                    if isinstance(result, Exception):
                        responses[agent_name] = {"status": "error", "error": str(result)}
                    else:
                        responses[agent_name] = result

        return responses

    async def _aggregate_responses(self, query: str, agent_responses: Dict[str, Any],
                                    supervision_result: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate responses from multiple agents"""
        aggregated_response = ""
        used_agents = []
        error_agents = []

        for agent_name, response in agent_responses.items():
            if response.get("status") == "success":
                used_agents.append(agent_name)
                response_text = response.get("response", "")
                if response_text:
                    aggregated_response += f"\n[{agent_name}]: {response_text}"
            else:
                error_agents.append(agent_name)

        if not aggregated_response:
            aggregated_response = "I processed your query but could not generate a detailed response. Please provide more context."

        return {
            "status": "success",
            "response": aggregated_response.strip(),
            "agent": "orchestrator",
            "agents_used": used_agents,
            "error_agents": error_agents,
            "supervision_routing": supervision_result.get("routing_decision", {}),
            "knowledge_context": supervision_result.get("knowledge_context", []),
            "metadata": {"total_agents_called": len(used_agents) + len(error_agents),
                         "successful_agents": len(used_agents), "failed_agents": len(error_agents)}
        }

    def _map_agent_name_to_type(self, agent_name: str) -> AgentType:
        mapping = {
            "developer_helper": AgentType.DEVELOPER_HELPER,
            "data_analyst": AgentType.DATA_ANALYST,
            "architect": AgentType.ARCHITECT,
            "documentation": AgentType.DOCUMENTATION,
            "supervisor": AgentType.SUPERVISOR
        }
        return mapping.get(agent_name, AgentType.DEVELOPER_HELPER)

    async def get_system_status(self) -> Dict[str, Any]:
        try:
            all_agents = self.agent_factory.get_all_agents()
            agent_statuses = {name: await agent.get_status() for name, agent in all_agents.items()}
            return {
                "status": "healthy",
                "total_queries_processed": self.total_queries,
                "average_processing_time": self.total_processing_time / max(self.total_queries, 1),
                "agent_call_counts": self.agent_calls,
                "agents": agent_statuses,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"status": "error", "message": str(e)}


_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator
