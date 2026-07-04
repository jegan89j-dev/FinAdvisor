"""Base agent interface for specialized financial agents."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from advisor.rag.retrieve import HybridRetriever


class BaseAgent(ABC):
    """Abstract base class for all specialized agents.
    
    Each agent:
    - Handles a specific financial domain
    - Has its own tool registry and prompts
    - Implements a run() method for single-turn execution
    - Can be composed into a multi-agent orchestrator
    """

    def __init__(self, agent_name: str, description: str):
        """Initialize the agent.
        
        Args:
            agent_name: Unique identifier (e.g., "risk_profiling")
            description: Human-readable description of agent responsibilities
        """
        self.agent_name = agent_name
        self.description = description
        self.max_steps = 6  # Inherited from original ReAct loop
        self.tool_output_limit = 4000  # chars

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool schema for this agent.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        pass

    @abstractmethod
    def get_system_prompt(self, profile: Optional[Dict[str, Any]] = None) -> str:
        """Return the system prompt for this agent.
        
        Args:
            profile: User profile dict (age, risk tolerance, goals, etc.)
            
        Returns:
            System prompt string tailored to this agent's domain
        """
        pass

    @abstractmethod
    def run(
        self,
        user_msg: str,
        history: List[Dict[str, str]],
        profile: Optional[Dict[str, Any]] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> str:
        """Execute the agent for a single user turn.
        
        Args:
            user_msg: Current user message
            history: Conversation history (prior user/assistant messages)
            profile: User profile for personalization
            retriever: Optional RAG retriever for grounding
            
        Returns:
            Agent's response string
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.agent_name!r})"
