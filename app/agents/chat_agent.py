"""
Simple conversational agent using LangChain

This is your first LangChain agent! It demonstrates:
1. How to initialize an LLM (Large Language Model)
2. How to create prompt templates
3. How to build chains using LCEL (LangChain Expression Language)
4. How to handle conversation history

Learning Resources:
- LangChain Docs: https://python.langchain.com/docs/
- LCEL Guide: https://python.langchain.com/docs/expression_language/
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import get_settings
import logging
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatAgent:
    """
    A simple conversational AI agent powered by LangChain
    
    This agent can:
    - Answer questions about SAP BTP
    - Maintain conversation context
    - Provide technical guidance
    
    Architecture:
        User Input → Prompt Template → LLM (GPT-4) → Output Parser → Response
    """
    
    def __init__(self):
        """
        Initialize the chat agent with LLM and prompt template
        
        Steps:
        1. Create LLM instance (the "brain")
        2. Define prompt template (the "instructions")
        3. Build chain using LCEL (the "pipeline")
        """
        logger.info("Initializing ChatAgent...")
        
        # STEP 1: Initialize the LLM
        # ---------------------------
        # ChatOpenAI is a wrapper around OpenAI's API
        # It handles API calls, retries, and response parsing
        # Use model from environment (OPENAI_MODEL), default to gpt-3.5-turbo if not set
        model_name = getattr(settings, 'openai_model', 'gpt-3.5-turbo')
        self.llm = ChatOpenAI(
            model=model_name,           # Model to use (configured via environment)
            temperature=0.7,             # Creativity: 0=focused, 1=creative
            api_key=settings.openai_api_key,
            max_tokens=1000,             # Maximum response length
            request_timeout=30           # Timeout for API calls
        )
        logger.info(f"LLM initialized: {model_name}")
        
        # STEP 2: Create Prompt Template
        # -------------------------------
        # Prompt template defines how we talk to the LLM
        # It includes:
        # - System message: Sets the AI's personality and role
        # - History placeholder: Where past messages go
        # - Human message: Current user input
        self.prompt = ChatPromptTemplate.from_messages([
            # System message: Define the agent's behavior and expertise
            ("system", self._get_system_prompt()),
            
            # Conversation history: Previous messages for context
            # This allows the agent to "remember" the conversation
            MessagesPlaceholder(variable_name="history"),
            
            # Current user input
            ("human", "{input}")
        ])
        logger.info("✓ Prompt template created")
        
        # STEP 3: Build Chain using LCEL
        # -------------------------------
        # LCEL (LangChain Expression Language) uses the pipe operator |
        # Think of it as: prompt → LLM → parser
        # 
        # self.prompt: Formats the input
        # self.llm: Sends to OpenAI and gets response
        # StrOutputParser(): Converts response to string
        self.chain = (
            self.prompt 
            | self.llm 
            | StrOutputParser()
        )
        logger.info("✓ Chain built using LCEL")
        
        # Statistics tracking
        self.total_requests = 0
        self.last_request_time = None
        
        logger.info("ChatAgent initialization complete! 🚀")
    
    def _get_system_prompt(self) -> str:
        """
        Define the system prompt that sets the agent's behavior
        
        This is crucial for:
        - Setting the AI's personality
        - Defining its expertise
        - Establishing boundaries
        
        Returns:
            str: System prompt text
        """
        return """You are an intelligent AI assistant specialized in SAP Business Technology Platform (BTP).

Your expertise includes:
- SAP BTP core concepts and architecture
- CAP (Cloud Application Programming) Model
- HANA Cloud database
- Cloud Foundry deployment
- Fiori applications
- OData services
- Authentication and authorization
- Integration patterns

Your communication style:
- Clear and concise explanations
- Use code examples when helpful
- Break down complex topics step-by-step
- Acknowledge when you're not certain about something
- Focus on practical, actionable advice

When helping with code:
- Explain what the code does
- Highlight best practices
- Suggest improvements when relevant
- Consider SAP-specific patterns and conventions

Remember: You're here to help developers learn and succeed with SAP BTP!"""
    
    async def get_response(
        self, 
        message: str, 
        history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get a response from the agent
        
        This is the main method that processes user input and returns a response.
        
        Flow:
        1. Convert history to LangChain message format
        2. Invoke the chain with user input and history
        3. Track statistics (time, tokens)
        4. Return structured response
        
        Args:
            message: User's input message
            history: List of previous messages [{"role": "user", "content": "..."}]
            
        Returns:
            Dict with response and metadata
            
        Example:
            >>> agent = ChatAgent()
            >>> result = await agent.get_response("What is CAP model?")
            >>> print(result["response"])
        """
        start_time = time.time()
        self.total_requests += 1
        self.last_request_time = start_time
        
        try:
            logger.info(f"Processing request #{self.total_requests}: {message[:50]}...")
            
            # STEP 1: Format conversation history
            # ------------------------------------
            # Convert from dict format to LangChain message objects
            formatted_history = []
            if history:
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    # Convert to appropriate message type
                    if role == "user":
                        formatted_history.append(HumanMessage(content=content))
                    elif role == "assistant":
                        formatted_history.append(AIMessage(content=content))
                    elif role == "system":
                        formatted_history.append(SystemMessage(content=content))
            
            logger.debug(f"Formatted {len(formatted_history)} history messages")
            
            # STEP 2: Invoke the chain
            # -------------------------
            # This is where the magic happens!
            # The chain will:
            # 1. Format the prompt with input and history
            # 2. Send to OpenAI API
            # 3. Parse the response
            response = await self.chain.ainvoke({
                "input": message,
                "history": formatted_history
            })
            
            # STEP 3: Calculate metrics
            # --------------------------
            response_time = time.time() - start_time
            
            logger.info(f"✓ Response generated in {response_time:.2f}s")
            
            # STEP 4: Return structured result
            # ---------------------------------
            return {
                "response": response,
                "response_time": response_time,
                "model": self.llm.model_name,
                "total_requests": self.total_requests
            }
            
        except Exception as e:
            logger.error(f"Error in ChatAgent.get_response: {str(e)}", exc_info=True)
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status
        
        Useful for health checks and monitoring
        
        Returns:
            Dict with agent status information
        """
        return {
            "agent_type": "chat",
            "status": "healthy",
            "model": self.llm.model_name,
            "total_requests": self.total_requests,
            "last_request_time": self.last_request_time
        }


# Example usage (for testing)
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        """Quick test of the agent"""
        agent = ChatAgent()
        
        # Test 1: Simple question
        result = await agent.get_response("What is SAP BTP?")
        print("Response:", result["response"])
        print(f"Time: {result['response_time']:.2f}s")
        
        # Test 2: With history
        history = [
            {"role": "user", "content": "What is CAP?"},
            {"role": "assistant", "content": "CAP is the Cloud Application Programming model..."}
        ]
        result = await agent.get_response("Can you explain more?", history=history)
        print("\nWith History:", result["response"])
    
    # Run test
    # asyncio.run(test_agent())