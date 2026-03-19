import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MockChatAgent:
    """
    Mock agent that returns predefined responses
    
    Used for testing when real OpenAI API is unavailable.
    """
    
    # Knowledge base of mock responses
    RESPONSES = {
        "sap btp": "SAP BTP (Business Technology Platform) is SAP's cloud platform offering. It provides a fully integrated set of cloud services including databases, analytics, app development, and integration capabilities. It's built on open technologies like Kubernetes and supports both SAP and non-SAP applications.",
        
        "cloud platform": "A cloud platform is an infrastructure for building, testing, and hosting applications on the internet. SAP BTP is SAP's cloud platform offering comprehensive services for enterprise applications.",
        
        "kubernetes": "Kubernetes is an open-source container orchestration platform that automates many of the manual processes involved in deploying, managing, and scaling containerized applications.",
        
        "database": "A database is an organized collection of structured data stored and accessed electronically. SAP BTP supports multiple database options including SAP HANA, PostgreSQL, and others.",
        
        "microservices": "Microservices is an architectural approach where applications are built as a collection of small, independent services that communicate over the network. It enables scalability and flexibility.",
        
        "api": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other and exchange data.",
        
        "integration": "Integration involves connecting different systems and applications to work together seamlessly. SAP BTP provides integration services to connect SAP and non-SAP systems.",
        
        "default": "I'm a mock AI agent for demonstration purposes. I can answer questions about SAP BTP, cloud platforms, and related technologies. Try asking about: SAP BTP, Cloud Platform, Kubernetes, Microservices, APIs, or Integration.",
    }
    
    def __init__(self):
        """Initialize the mock agent"""
        logger.info("Initializing Mock Chat Agent (demo mode)...")
        self.request_count = 0
        logger.info("✓ Mock Chat Agent ready (no API calls will be made)")
    
    async def get_response(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Get a mock response for the given message
        
        Args:
            message: User's question
            history: Optional conversation history
            
        Returns:
            Dict with response, model, and timing info
        """
        self.request_count += 1
        start_time = datetime.utcnow()
        
        logger.info(f"Processing mock request #{self.request_count}: {message[:50]}...")
        
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        # Find matching response
        message_lower = message.lower()
        response_text = self.RESPONSES.get("default")
        
        for keyword, response in self.RESPONSES.items():
            if keyword in message_lower:
                response_text = response
                break
        
        # Add context from history if available
        if history and len(history) > 0:
            response_text = f"Based on our conversation, here's my response:\n\n{response_text}"
        
        # Calculate response time
        end_time = datetime.utcnow()
        response_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✓ Mock response generated (time: {response_time:.2f}s)")
        
        return {
            "response": response_text,
            "model": "mock-agent",
            "response_time": response_time,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_type": "mock",
            "status": "healthy",
            "model": "mock-agent",
            "last_request_time": datetime.utcnow().isoformat(),
            "total_requests": self.request_count,
        }
