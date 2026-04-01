import logging
import re
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ToolStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any]
    required_params: List[str]
    agent_types: List[str]
    status: ToolStatus = ToolStatus.AVAILABLE


class ToolsRegistry:
    """Registry of tools available to agents"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_all_tools()
        logger.info(f"Tools registry initialized with {len(self.tools)} tools")

    def _register_all_tools(self):
        self._register_code_analyzer()
        self._register_architecture_visualizer()
        self._register_dependency_graph()
        self._register_sql_query_builder()
        self._register_data_profiler()
        self._register_system_analyzer()
        self._register_integration_mapper()
        self._register_doc_generator()
        self._register_diagram_creator()
        self._register_knowledge_search()
        self._register_conversation_retriever()

    def _register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def _register_code_analyzer(self):
        async def analyze_code(code: str, language: str = "python") -> Dict[str, Any]:
            lines = code.split('\n')
            functions = len(re.findall(r'def\s+\w+', code))
            classes = len(re.findall(r'class\s+\w+', code))
            imports = len(re.findall(r'^(import|from)\s+', code, re.MULTILINE))
            return {"status": "success", "language": language, "total_lines": len(lines),
                    "functions": functions, "classes": classes, "imports": imports,
                    "complexity": "medium" if functions > 5 else "low"}

        self._register_tool(Tool(
            name="analyze_code", description="Analyze source code structure and patterns",
            func=analyze_code, parameters={"code": {"type": "string"}, "language": {"type": "string"}},
            required_params=["code"], agent_types=["developer_helper"]))

    def _register_architecture_visualizer(self):
        async def visualize_architecture(components: List[str], relationships: List[Dict]) -> Dict[str, Any]:
            diagram = "```\n" + "\n".join(f"[{c}]" for c in components) + "\n```"
            return {"status": "success", "components_count": len(components),
                    "relationships_count": len(relationships), "diagram": diagram}

        self._register_tool(Tool(
            name="visualize_architecture", description="Create architecture diagrams",
            func=visualize_architecture,
            parameters={"components": {"type": "array"}, "relationships": {"type": "array"}},
            required_params=["components"], agent_types=["developer_helper", "architect"]))

    def _register_dependency_graph(self):
        async def generate_dependency_graph(components: List[str]) -> Dict[str, Any]:
            return {"status": "success", "components": len(components), "graph_type": "directed_acyclic_graph"}

        self._register_tool(Tool(
            name="generate_dependency_graph", description="Generate component dependency graphs",
            func=generate_dependency_graph, parameters={"components": {"type": "array"}},
            required_params=["components"], agent_types=["developer_helper", "architect"]))

    def _register_sql_query_builder(self):
        async def build_sql_query(table: str, filters: Optional[Dict] = None, columns: Optional[List[str]] = None) -> Dict[str, Any]:
            selected_columns = ", ".join(columns) if columns else "*"
            query = f"SELECT {selected_columns} FROM {table}"
            if filters:
                where_clauses = [f"{k}='{v}'" for k, v in filters.items()]
                query += " WHERE " + " AND ".join(where_clauses)
            return {"status": "success", "query": query, "table": table}

        self._register_tool(Tool(
            name="build_sql_query", description="Build SQL queries for data analysis",
            func=build_sql_query,
            parameters={"table": {"type": "string"}, "filters": {"type": "object"}, "columns": {"type": "array"}},
            required_params=["table"], agent_types=["data_analyst"]))

    def _register_data_profiler(self):
        async def profile_data(table: str, sample_size: int = 1000) -> Dict[str, Any]:
            return {"status": "success", "table": table, "sample_size": sample_size}

        self._register_tool(Tool(
            name="profile_data", description="Profile data in tables",
            func=profile_data, parameters={"table": {"type": "string"}, "sample_size": {"type": "integer"}},
            required_params=["table"], agent_types=["data_analyst"]))

    def _register_system_analyzer(self):
        async def analyze_system(system_name: str) -> Dict[str, Any]:
            return {"status": "success", "system": system_name}

        self._register_tool(Tool(
            name="analyze_system", description="Analyze system architecture",
            func=analyze_system, parameters={"system_name": {"type": "string"}},
            required_params=["system_name"], agent_types=["architect"]))

    def _register_integration_mapper(self):
        async def map_integrations(app: str) -> Dict[str, Any]:
            return {"status": "success", "application": app, "integrations": []}

        self._register_tool(Tool(
            name="map_integrations", description="Map application integrations",
            func=map_integrations, parameters={"app": {"type": "string"}},
            required_params=["app"], agent_types=["architect"]))

    def _register_doc_generator(self):
        async def generate_documentation(subject: str, doc_type: str = "markdown") -> Dict[str, Any]:
            return {"status": "success", "subject": subject, "format": doc_type}

        self._register_tool(Tool(
            name="generate_documentation", description="Generate documentation",
            func=generate_documentation,
            parameters={"subject": {"type": "string"}, "doc_type": {"type": "string"}},
            required_params=["subject"], agent_types=["documentation"]))

    def _register_diagram_creator(self):
        async def create_diagram(diagram_type: str, title: str, elements: List[str]) -> Dict[str, Any]:
            return {"status": "success", "type": diagram_type, "title": title, "elements": len(elements)}

        self._register_tool(Tool(
            name="create_diagram", description="Create architecture and data flow diagrams",
            func=create_diagram,
            parameters={"diagram_type": {"type": "string"}, "title": {"type": "string"}, "elements": {"type": "array"}},
            required_params=["diagram_type", "title"], agent_types=["documentation", "architect"]))

    def _register_knowledge_search(self):
        async def search_knowledge(query: str, limit: int = 5) -> Dict[str, Any]:
            from app.knowledge.knowledge_base import get_knowledge_base
            kb = get_knowledge_base()
            results = kb.search(query, k=limit)
            return {"status": "success", "query": query, "results_count": len(results), "results": results[:limit]}

        self._register_tool(Tool(
            name="search_knowledge", description="Search knowledge base",
            func=search_knowledge, parameters={"query": {"type": "string"}, "limit": {"type": "integer"}},
            required_params=["query"],
            agent_types=["supervisor", "developer_helper", "data_analyst", "architect", "documentation"]))

    def _register_conversation_retriever(self):
        async def retrieve_conversation_context(conversation_id: str, max_messages: int = 5) -> Dict[str, Any]:
            from app.knowledge.document_store import DocumentStore
            store = DocumentStore()
            conversation = store.get_conversation(conversation_id)
            if conversation:
                messages = conversation.messages[-max_messages:]
                return {"status": "success", "conversation_id": conversation_id,
                        "messages": [{"role": m.role, "content": m.content} for m in messages]}
            return {"status": "not_found"}

        self._register_tool(Tool(
            name="retrieve_conversation_context", description="Retrieve conversation context",
            func=retrieve_conversation_context,
            parameters={"conversation_id": {"type": "string"}, "max_messages": {"type": "integer"}},
            required_params=["conversation_id"],
            agent_types=["supervisor", "developer_helper", "data_analyst", "architect", "documentation"]))

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def get_tools_for_agent(self, agent_type: str) -> List[Tool]:
        return [tool for tool in self.tools.values() if agent_type in tool.agent_types]

    def get_all_tools(self) -> Dict[str, Tool]:
        return self.tools

    def get_tools_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {"description": tool.description, "parameters": tool.parameters,
                    "required_params": tool.required_params, "agent_types": tool.agent_types,
                    "status": tool.status.value}
            for name, tool in self.tools.items()
        }


_tools_registry: Optional[ToolsRegistry] = None


def get_tools_registry() -> ToolsRegistry:
    global _tools_registry
    if _tools_registry is None:
        _tools_registry = ToolsRegistry()
    return _tools_registry
